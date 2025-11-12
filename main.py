from musetalk import Inference

if __name__ == "__main__":
    inference = Inference(
        unet_model_path="/home/sergiukopcsa/Work/MuseTalk/models/musetalkV15/musetalkV15/unet.pth",
        unet_model_config="/home/sergiukopcsa/Work/MuseTalk/models/musetalkV15/musetalkV15/musetalk.json",
        checkpoints_dirs="/home/sergiukopcsa/Work/MuseTalk/models",
    )

    images = ["/home/sergiukopcsa/Work/MuseTalk/assets/demo/musk/musk.png"]
    inference.run(audio_path="/mnt/x/data_content/audio_sergiu_potato/love_potato.wav",images=images,  fps=25)