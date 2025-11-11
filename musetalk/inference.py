# musetalk/inference.py
from pathlib import Path
import yaml
import os
import cv2
import copy
import torch

import glob
import shutil
import pickle
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import WhisperModel
import sys

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

class Inference:
    """
    High-level inference wrapper for MuseTalk.
    """

    def __init__(
        self,
        unet_model_path: str,
        unet_model_config: str,
        whisper_dir: str,
        vae_type: str ="sd-vae",
        use_float16 = False,
        left_cheek_width=90,
        right_cheek_width=90,
    ):

        self._device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            self._vae, self._unet, self._pe = load_all_model(
                unet_model_path=unet_model_path,
                vae_type=vae_type,
                unet_config=unet_model_config,
                device=self._device
            )

            if use_float16:
                self._pe = self._pe.half()
                self._vae.vae = self._vae.vae.half()
                self._unet.model = self._unet.model.half()


            self._pe = self._pe.to(self._device)
            self._vae.vae = self._vae.vae.to(self._device)
            self._unet.model = self._unet.model.to(self._device)


            self._audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
            self._weight_dtype = self._unet.model.dtype
            self._whisper = WhisperModel.from_pretrained(whisper_dir)
            self._whisper = self._whisper.to(device=self._device, dtype=self._weight_dtype).eval()
            self._whisper.requires_grad_(False)

            self._fp = FaceParsing(
                left_cheek_width=left_cheek_width,
                right_cheek_width=right_cheek_width
            )

    def run(self, images, audio_path, fps, audio_padding_length_left=2, audio_padding_length_right=2, extra_margin=10, batch_size=8, parsing_mode="jaw"):
        with torch.no_grad():
            whisper_input_features, librosa_length = self._audio_processor.get_audio_feature(audio_path)
            whisper_chunks = self._audio_processor.get_whisper_chunk(
                whisper_input_features, 
                self._device, 
                self._weight_dtype, 
                self._whisper, 
                librosa_length,
                fps=fps,
                audio_padding_length_left=audio_padding_length_left,
                audio_padding_length_right=audio_padding_length_right,
            )

            coord_list, frame_list = get_landmark_and_bbox(images, 0)

            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                y2 = y2 + extra_margin
                y2 = min(y2, frame.shape[0])
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
                latents = self._vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)
        
    
            
            # Smooth first and last frames
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            
            # Batch inference
            video_num = len(whisper_chunks)
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
                device=self._device,
            )
            
            res_frame_list = []
            total = int(np.ceil(float(video_num) / batch_size))


            timesteps = torch.tensor([0], device=self._device)
            
            # Execute inference
            for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                audio_feature_batch = self._pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=self._unet.model.dtype)
                
                pred_latents = self._unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = self._vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)
            
            # Pad generated images to original size
            print("Padding generated images to original video size")
            results = []
            for i, res_frame in enumerate(tqdm(res_frame_list)):
                bbox = coord_list_cycle[i%(len(coord_list_cycle))]
                ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
                x1, y1, x2, y2 = bbox
                y2 = y2 + extra_margin
                y2 = min(y2, frame.shape[0])
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                
                combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=parsing_mode, fp=self._fp)
                results.append(combine_frame)
            return results
