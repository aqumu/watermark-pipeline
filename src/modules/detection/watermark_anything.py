import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from PIL import Image

from src.utils.logger import logger
from ..base import BaseDetector


class WatermarkAnythingDetector(BaseDetector):
    def __init__(
        self,
        model_root: str = "external/watermark-anything",
        checkpoint_url: str = "https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth",
        checkpoint_path: str = "checkpoints/wam_mit.pth",
        params_path: str = "checkpoints/params.json",
        mask_threshold: float = 0.5,
        use_gpu: bool = False,
    ):
        self.model_root = Path(model_root).resolve()
        self.checkpoint_url = checkpoint_url
        self.checkpoint_path = self.model_root / checkpoint_path
        self.params_path = self.model_root / params_path
        self.mask_threshold = mask_threshold
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self._ensure_repo()

        if str(self.model_root) not in sys.path:
            sys.path.insert(0, str(self.model_root))

        from watermark_anything.augmentation.augmenter import Augmenter
        from watermark_anything.data.transforms import default_transform, normalize_img, unnormalize_img
        from watermark_anything.models import Wam, build_embedder, build_extractor
        from watermark_anything.modules.jnd import JND

        self.default_transform = default_transform
        self.normalize_img = normalize_img
        self.unnormalize_img = unnormalize_img
        self.Wam = Wam
        self.build_embedder = build_embedder
        self.build_extractor = build_extractor
        self.Augmenter = Augmenter
        self.JND = JND

        self.model = self._load_model().to(self.device).eval()
        logger.info(f"Initialized WatermarkAnythingDetector on {self.device.type}")

    def _ensure_repo(self) -> None:
        if self.model_root.exists():
            return

        self.model_root.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cloning Watermark Anything into {self.model_root}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/facebookresearch/watermark-anything", str(self.model_root)],
            check=True,
        )

    def _load_model(self):
        if not self.params_path.exists():
            raise FileNotFoundError(f"Watermark Anything params not found: {self.params_path}")

        if not self.checkpoint_path.exists():
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading Watermark Anything checkpoint to {self.checkpoint_path}")
            urlretrieve(self.checkpoint_url, self.checkpoint_path)

        with open(self.params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        args = argparse.Namespace(**params)

        embedder_cfg = omegaconf.OmegaConf.load(self.model_root / args.embedder_config)
        extractor_cfg = omegaconf.OmegaConf.load(self.model_root / args.extractor_config)
        augmenter_cfg = omegaconf.OmegaConf.load(self.model_root / args.augmentation_config)
        attenuation_cfg = omegaconf.OmegaConf.load(self.model_root / args.attenuation_config)

        embedder = self.build_embedder(args.embedder_model, embedder_cfg[args.embedder_model], args.nbits)
        extractor = self.build_extractor(
            extractor_cfg.model,
            extractor_cfg[args.extractor_model],
            args.img_size,
            args.nbits,
        )
        augmenter = self.Augmenter(**augmenter_cfg)

        attenuation = None
        attenuation_name = getattr(args, "attenuation", None)
        if attenuation_name and attenuation_name in attenuation_cfg:
            attenuation = self.JND(
                **attenuation_cfg[attenuation_name],
                preprocess=self.unnormalize_img,
                postprocess=self.normalize_img,
            )

        model = self.Wam(
            embedder,
            extractor,
            augmenter,
            attenuation=attenuation,
            scaling_w=args.scaling_w,
            scaling_i=args.scaling_i,
            roll_probability=getattr(args, "roll_probability", 0),
            img_size_extractor=getattr(args, "img_size_extractor", args.img_size),
        )

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
        return model

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> tuple[np.ndarray, bool]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pt = self.default_transform(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)

        preds = self.model.detect(image_pt)["preds"]
        mask_logits = preds[:, 0:1, :, :]
        mask_probs = torch.sigmoid(mask_logits)
        mask_probs = F.interpolate(
            mask_probs,
            size=(image.shape[0], image.shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        mask_binary = (mask_probs.squeeze().cpu().numpy() >= self.mask_threshold).astype(np.uint8) * 255
        detected = bool(np.any(mask_binary > 0))
        return mask_binary, detected
