# musetalk/inference.py
from pathlib import Path
import yaml

class Inference:
    """
    High-level inference wrapper for MuseTalk.
    """

    def __init__(
        self,
        config_path: str = "configs/inference/test.yaml",
        audio_path: str | None = None,
        video_path: str | None = None,
        output_dir: str = "outputs"
    ):
        self.config_path = Path(config_path)
        self.audio_path = Path(audio_path) if audio_path else None
        self.video_path = Path(video_path) if video_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def run(self):
        """
        Run the inference pipeline.
        Replace this stub with actual logic or import scripts.inference
        """
        print(f"[MuseTalk] Running inference with config: {self.config_path}")
        print(f"Audio: {self.audio_path}")
        print(f"Video: {self.video_path}")
        print(f"Output directory: {self.output_dir}")

        # Example: you could call your actual inference script
        # from scripts.inference import run_inference
        # return run_inference(self.config, self.audio_path, self.video_path)

        result_path = self.output_dir / "result.mp4"
        print(f"✅ Inference complete. Saved result to {result_path}")
        return result_path
