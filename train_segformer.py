#!/usr/bin/env python3
"""
Train and evaluate a SegFormer-based binary segmentation pipeline for industrial defect
detection on multi-channel FSCAN data.

Main features
-------------
- multi-seed experiments with automatic seed discovery
- multi-channel input construction from physically-derived FSCAN channels
- optional FFT-based low-pass / residual preprocessing
- SegFormer input adaptation for non-RGB inputs
- optional CutMix augmentation in channel space
- checkpointing, overlay generation, and aggregated evaluation

Notes
-----
This script is intentionally kept as a single file for reproducibility and ease of inspection.
The raw industrial dataset is not public, so the repository is intended to document the
training and evaluation pipeline rather than provide a runnable benchmark out of the box.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

# ==================================================
#                      CONFIG
# ==================================================
@dataclass
class CFG:

    # ==================================================
    #   DATASET    
    #   - seeds_root
    # ==================================================
    seeds_root: str = "data/seeds"
    seed_glob: str = "SEED_*"              # seed folder pattern
    fscan_subdir: str = "fscans"           # inside train/ and valid/
    masks_subdir: str = "masks"            # inside train/ and valid/

    # Number of defect classes in indexed masks (excluding background)
    num_defect_classes: int = 12

    # ==================================================
    # Input representation (FSCAN stacking)
    # ==================================================
    # Which channels to stack as model input, in order.
    # Can include repeats, e.g. ("fibular","fibular","fibular") -> in_channels=3.
    input_streams: Tuple[str, ...] = ("azimuthal", "fibular", "diffuse", "specular")

    # Optional FFT processing per stream:
    #   "None" | "lpf" | "residual"
    # Examples:
    #   {"diffuse": "lpf"}
    #   {"fibular": "lpf", "diffuse": "lpf"}
    #   {"diffuse": "residual"}
    fft_mode: Optional[Dict[str, str]] = field(default_factory=lambda: {"fibular": "lpf", "diffuse": "residual"})
    fft_lp_radius: int = 30               # radius for low-pass FFT mask

    # Normalization behavior:
    # - If True: min/max normalize each channel after mapping/FFT.
    # - If False: only azimuthal is mapped to [0,1], others are left as-is.
    use_minmax_norm: bool = True

    # Optional ImageNet normalization applied AFTER the above preprocessing.
    # Only valid when in_channels == 3.
    imagenet_norm: bool = False
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ==================================================
    # Visualization / overlays
    # ==================================================
    # Which single stream (must be present in input_streams) to use as overlay background.
    overlay_stream: str = "fibular"
    vis_alpha: float = 0.45               # transparency of overlay masks
    max_vis: int = 200                    # max saved overlays per evaluation call

    # Dump a small subset of model inputs to disk for sanity-checking preprocessing
    # and data loading (after all mappings, FFT, normalization, and resizing).
    dump_inputs: bool = True
    # Maximum number of input samples to dump per seed/run.
    # Samples are taken from the first batches of the training DataLoader.
    dump_inputs_max: int = 25
    # Name of the subdirectory (inside each seed output folder) where dumped
    # input images are saved.
    dump_inputs_dirname: str = "inputs_dump"


    # ==================================================
    # Model (SegFormer)
    # ==================================================
    hf_ckpt: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    drop_path_rate: float = 0.0

    # When in_channels != 3, input projection must be adapted.
    # "mean_rgb": initialize new channels with mean of pretrained RGB weights
    # "xavier": random Xavier init
    inproj_init: str = "mean_rgb"

    # Optional initialization from an existing checkpoint.
    # If set, only the model weights (checkpoint["model"]) are loaded; optimizer state is NOT restored.
    init_pt: str = ""
    # Optional full training resume checkpoint.
    # If set, restores model weights, optimizer state, AMP scaler, and epoch counter,
    # allowing training to continue exactly from where it was interrupted.
    resume_pt: str = ""   # if set, resume optimizer+scaler+epoch

    # ==================================================
    # Training
    # ==================================================
    device: str = "auto"
    epochs: int = 100
    batch_size: int = 8

    lr: float = 4e-5
    weight_decay: float = 1e-4
    num_workers: int = 8
    use_amp: bool = True

    # Threshold for hard prediction (defect vs background)
    sel_thr: float = 0.5
    img_size: int = 512 
    # Save periodic checkpoints every N epochs (0 disables)
    ckpt_every: int = 0

    # ==================================================
    # Loss (Focal + Tversky)
    # ==================================================
    alpha_focal: float = 0.75   # <0 disables alpha weighting in focal
    gamma_focal: float = 1.75
    wf: float = 0.40            # mix: wf*focal + (1-wf)*tversky
    alpha_tv: float = 0.2       # weight for FP in Tversky denom
    beta_tv: float = 0.8        # weight for FN in Tversky denom
    # ==================================================
    # Reproducibility
    # ==================================================
    deterministic: bool = False

    # ==================================================
    # Output
    # ==================================================
    out_dir: str = "outputs"
    run_name: str = "segformer_fscan" 

    # ==================================================
    # Mode
    # ==================================================
    # Operating mode:
    # - "train": trains one model per seed and saves the best checkpoint (best.pt) in each seed folder.
    # - "eval": runs evaluation only (no training).
    mode: str = "train"       
    # Checkpoint selection for evaluation:
    # - If empty (""): each seed is evaluated using its own best.pt found in
    #   <out_dir>/<run_name>/<SEED_xxx>/best.pt.
    # - If set to a valid checkpoint path: the SAME checkpoint is evaluated across ALL seeds,
    #   ignoring per-seed best.pt files.
    # Used only when mode == "eval".
    ckpt_path: str = ""         # required only for mode="eval"

    # ==================================================
    # CutMix 
    # ==================================================
    cutmix_enable: bool = False
    cutmix_clean_fscans_dir: str = ""     # folder with clean *.fscan/*.h5
    cutmix_cleans_per_defect: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate SegFormer on multi-channel FSCAN data.")
    parser.add_argument("--seeds-root", type=str, default=None, help="Root folder containing SEED_* splits.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for run artifacts.")
    parser.add_argument("--run-name", type=str, default=None, help="Name of the output subdirectory.")
    parser.add_argument("--device", type=str, default=None, help="Device override: auto, cuda, cuda:0, cpu.")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default=None, help="Execution mode.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Checkpoint path used only in eval mode.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    return parser.parse_args()


def build_cfg_from_args() -> CFG:
    cfg = CFG()
    args = parse_args()
    if args.seeds_root is not None:
        cfg.seeds_root = args.seeds_root
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.device is not None:
        cfg.device = args.device
    if args.mode is not None:
        cfg.mode = args.mode
    if args.ckpt_path is not None:
        cfg.ckpt_path = args.ckpt_path
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    return cfg


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def cuda_sync() -> None:
    if torch.cuda.is_available():
        cuda_sync()


def validate_cfg(cfg: CFG) -> None:
    # imagenet_norm => exactly 3 streams
    if cfg.imagenet_norm and len(cfg.input_streams) != 3:
        raise ValueError(
            f"imagenet_norm=True requires len(input_streams)==3, got {len(cfg.input_streams)}: {cfg.input_streams}"
        )

    # overlay_stream must exist in inputs
    if cfg.overlay_stream not in cfg.input_streams:
        raise ValueError(
            f"overlay_stream='{cfg.overlay_stream}' must be in input_streams={cfg.input_streams}"
        )

    # FFT sanity: dict values must be allowed, and radius must be >0 if any FFT is active
    allowed = {"none", "lpf", "residual"}
    fft_map = cfg.fft_mode or {}
    for k, v in fft_map.items():
        if v not in allowed:
            raise ValueError(f"fft_mode['{k}']='{v}' invalid. Allowed: {sorted(allowed)}")

    any_fft = any(v in {"lpf", "residual"} for v in fft_map.values())
    if any_fft and cfg.fft_lp_radius <= 0:
        raise ValueError(f"fft_lp_radius must be > 0 when FFT is enabled, got {cfg.fft_lp_radius}")

    # eval semantics: if ckpt_path is set, it must exist
    if cfg.mode.lower() == "eval" and cfg.ckpt_path:
        p = Path(cfg.ckpt_path)
        if not p.is_file():
            raise FileNotFoundError(f"cfg.ckpt_path not found or not a file: {p}")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEED_DIR_RE = re.compile(r"SEED_(\d+)") # SEED_ is the literal prefix, and (\d+) means to capture one or more digits

# ==================================================
#     Model Adaptation (input projection layer)
# ==================================================
def adapt_segformer_input_channels(model, in_ch: int, init: str = "mean_rgb"):
    proj = model.segformer.encoder.patch_embeddings[0].proj
    old_w = proj.weight.data
    out_ch, old_in, kh, kw = old_w.shape
    if old_in == in_ch:
        return

    new_proj = nn.Conv2d(in_ch, out_ch, (kh, kw), stride=proj.stride, padding=proj.padding, bias=(proj.bias is not None))

    with torch.no_grad():
        if init == "mean_rgb":
            # if old_in >= 1, average across existing channels and repeat
            mean_w = old_w.mean(dim=1, keepdim=True)
            new_proj.weight.copy_(mean_w.repeat(1, in_ch, 1, 1))
        elif init == "xavier":
            nn.init.xavier_uniform_(new_proj.weight)
        else:
            raise ValueError(init)

        if proj.bias is not None:
            new_proj.bias.copy_(proj.bias.data)

    model.segformer.encoder.patch_embeddings[0].proj = new_proj

def build_model(cfg: CFG):
    in_ch = len(cfg.input_streams)

    # 1) Load true pretrained weights in the 3-channel configuration
    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg.hf_ckpt,
        num_labels=1,
        drop_path_rate = cfg.drop_path_rate,
        ignore_mismatched_sizes=True,
    )


    # 3) Adapt input projection AFTER pretrained load
    adapt_segformer_input_channels(model, in_ch=in_ch, init=cfg.inproj_init)
    # 4) Update model config to reflect real input channels
    model.config.num_channels = in_ch

    # 5) Optional init from your checkpoint
    if cfg.init_pt:
        ckpt = torch.load(cfg.init_pt, map_location=cfg.device)
        sd = ckpt["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
       
        print(f"[INIT] Missing={len(missing)} Unexpected={len(unexpected)}")

    return model
# ==================================================
#       Loss Function
# ==================================================
class FocalTverskyLoss(nn.Module):
    """
    loss = wf * focal_mean  +  (1 - wf) * (1 - TP / (TP + alpha_tv*FP + beta_tv*FN + eps))

    - Logits:  [B,1,H,W]  (single-channel, positive class)
    - Targets: [B,1,H,W] or [B,H,W] with {0,1}
    - Focal:   computed with reduction='none' then pixel-wise mean (scale-free)
    - Tversky: TP/FP/FN summed globally over the batch (scale-free)
    """
    def __init__(
        self,
        alpha_focal: float = -1.0,  # set <0 to disable class weighting (alpha=None)
        gamma_focal: float = 1.5,
        wf: float = 0.10,
        alpha_tv: float = 0.3,
        beta_tv: float = 0.7,
        eps: float = 1e-7,
    ):
        super().__init__()
        assert 0.0 <= wf <= 1.0
        self.alpha_focal = float(alpha_focal)
        self.gamma_focal = float(gamma_focal)
        self.wf = float(wf)
        self.alpha_tv = float(alpha_tv)
        self.beta_tv = float(beta_tv)
        self.eps = float(eps)

    @staticmethod
    def _to_bhw(x: torch.Tensor) -> torch.Tensor:
        # [B,1,H,W] -> [B,H,W]; pass-through if already [B,H,W]
        return x[:, 0] if (x.dim() == 4 and x.size(1) == 1) else x

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = (targets > 0).to(dtype=torch.float32)

        # FOCAL: pixel-wise mean over the whole batch
        focal_mean = sigmoid_focal_loss(
            inputs=logits.float(),
            targets=targets,
            alpha=None if self.alpha_focal < 0 else self.alpha_focal,
            gamma=self.gamma_focal,
            reduction="mean",
        )  # scalar

        # TVERSKY: global over the batch
        p = torch.sigmoid(logits.float())   # [B,1,H,W]
        p = self._to_bhw(p)                 # [B,H,W]
        t = self._to_bhw(targets)           # [B,H,W]

        tp = (p * t).sum()
        fp = (p * (1.0 - t)).sum()
        fn = ((1.0 - p) * t).sum()

       
        denom = tp + self.alpha_tv * fp + self.beta_tv * fn + self.eps
        tversky_loss = 1.0 - tp / denom
        if not torch.isfinite(tversky_loss):
            raise RuntimeError(
                f"Tversky became non-finite: tp={tp.item()}, fp={fp.item()}, fn={fn.item()}, denom={denom.item()}"
            )


        return self.wf * focal_mean + (1.0 - self.wf) * tversky_loss

# ==================================================
#    Data Normalization / FFT
# ==================================================
def azimuthal_to_01(az: np.ndarray) -> np.ndarray:
    """Map azimuthal from [-pi, pi] to [0,1]."""
    az = az.astype(np.float32)
    return (az + np.pi) / (2.0 * np.pi)

def lowpass_mask(shape: tuple[int, int], radius: int) -> np.ndarray:
    """Circular low-pass mask (1=keep low freq, 0=remove high freq)."""
    H, W = shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return (dist <= float(radius)).astype(np.float32)

def fft_lowpass_residual(img: np.ndarray, lp_radius: int) -> np.ndarray:
    """
    Compute residual = img - lowpass(img) using FFT masking.
    img must be float32 2D.
    """
    img = img.astype(np.float32)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    mask = lowpass_mask(img.shape, lp_radius)
    fshift_lp = fshift * mask

    img_lp = np.fft.ifft2(np.fft.ifftshift(fshift_lp))
    img_lp = np.real(img_lp).astype(np.float32)

    residual = img - img_lp
    return residual

def fft_lowpass_only(img: np.ndarray, lp_radius: int) -> np.ndarray:
    """
    Compute low-pass filtered image using FFT masking.
    img must be float32 2D.
    """
    img = img.astype(np.float32)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    mask = lowpass_mask(img.shape, lp_radius)
    fshift_lp = fshift * mask

    img_lp = np.fft.ifft2(np.fft.ifftshift(fshift_lp))
    img_lp = np.real(img_lp).astype(np.float32)

    return img_lp

def read_fscan_channels(fscan_path: Path, keys: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    """
    Reads multiple 2D channels from a single fscan/h5 file.

    - keys: requested channel names, e.g. ("azimuthal","fibular","diffuse")
    - Returns dict: key -> float32 2D array.
    - Ignores any other keys (e.g. "raw", "pixpermm") because we never request them.
    """
    out: Dict[str, np.ndarray] = {}
    with h5py.File(fscan_path, "r") as f:
        available = set(f.keys())
        for k in keys:
            if k not in available:
                raise KeyError(
                    f"Key '{k}' not found in {fscan_path}. Available keys: {sorted(list(available))}"
                )
            arr = np.array(f[k], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array for key '{k}', got shape {arr.shape} in {fscan_path}")
            out[k] = arr
    return out

def minmax_norm_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx <= mn + eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)

def physical_map_channel(name: str, img: np.ndarray) -> np.ndarray:
    """
    Only apply mandatory physical mapping, no per-image normalization.
    - azimuthal: map [-pi, pi] -> [0,1]
    - others: return float32 as-is
    """
    img = img.astype(np.float32)
    if name == "azimuthal":
        return azimuthal_to_01(img)
    return img

def symm_residual_to_01(r: np.ndarray, p: float = 99.0, eps: float = 1e-8) -> np.ndarray:
    """
    Map residual (can be negative) to [0,1] while preserving sign information.

    Steps:
      1) s = percentile(|r|, p)
      2) r_scaled = clip(r / s, -1, 1)
      3) map to [0,1] via 0.5 + 0.5*r_scaled
    """
    r = r.astype(np.float32)
    s = np.percentile(np.abs(r), p)
    s = float(s) if np.isfinite(s) and s > eps else 1.0
    r_scaled = np.clip(r / s, -1.0, 1.0)
    return (0.5 + 0.5 * r_scaled).astype(np.float32)

def apply_fft(img: np.ndarray, mode: str, lp_radius: int) -> np.ndarray:
    if mode is None or mode == "none":
        return img
    if mode == "lpf":
        return fft_lowpass_only(img, lp_radius)
    if mode == "residual":
        return fft_lowpass_residual(img, lp_radius)
    raise ValueError(f"Unknown fft mode: {mode}")

def build_input_stack_from_fscan(
    fscan_path: Path,
    streams: Tuple[str, ...],
    fft_mode: Optional[Dict[str, str]],
    lp_radius: int,
    use_minmax_norm: bool,
    imagenet_norm: bool,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    residual_percentile: float = 99.0,  
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      x_final: [C,H,W] float32 (possibly ImageNet normalized)
      x_for_vis: [C,H,W] float32 in a human-friendly space (pre-ImageNet)
    """
    uniq = tuple(dict.fromkeys(streams).keys())
    data = read_fscan_channels(fscan_path, uniq)

    chans_vis = []
    chans_final = []

    for s in streams:
        raw = data[s]

        # 1) Mandatory physical mapping only (no minmax yet)
        img = physical_map_channel(s, raw)   # azimuthal -> [0,1], others float32 raw

        # Save a “visualizable” version BEFORE imagenet norm
        chans_vis.append(img.astype(np.float32))

        # 2) FFT on raw/mapped (not minmax)
        mode = "none" if not fft_mode else fft_mode.get(s, "none")
        if mode != "none":
            img_fft = apply_fft(img, mode, lp_radius)

            # 3) Post-FFT scaling
            if mode == "residual":
                # preserve sign, produce [0,1] representation
                img_scaled = symm_residual_to_01(img_fft, p=residual_percentile)
            else:
                # lpf output: choose whether to scale
                if use_minmax_norm:
                    img_scaled = minmax_norm_01(img_fft)
                else:
                    img_scaled = img_fft.astype(np.float32)
        else:
            # 3) No FFT: final scaling depends on use_minmax_norm
            if use_minmax_norm:
                # azimuthal already [0,1] but minmax is OK; others get minmax
                img_scaled = minmax_norm_01(img)
            else:
                img_scaled = img.astype(np.float32)


        # 4) For model input: in your current design you want the same as vis,
        #    then optionally ImageNet normalize at the end (only for 3ch)
        chans_final.append(img_scaled.astype(np.float32))

    x_vis = np.stack(chans_vis, axis=0).astype(np.float32)     # [C,H,W]
    x     = np.stack(chans_final, axis=0).astype(np.float32)   # [C,H,W]
    # 5) ImageNet normalization (only meaningful for C==3)
    if imagenet_norm:
        if x.shape[0] != 3:
            raise ValueError("imagenet_norm=True requires 3 input channels")
        mean_arr = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        std_arr  = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        x = (x - mean_arr) / std_arr

    return x, x_vis

# ==================================================
#    Reproducibility
# ==================================================
def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # More reproducible, slower
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        # Faster, slightly less reproducible
        torch.backends.cudnn.benchmark = True

def per_donor_seed(global_seed: int, donor_stem: str) -> int:
    import hashlib
    h = hashlib.sha1(f"{global_seed}::{donor_stem}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def make_worker_init_fn(seed: int):
    def _init_fn(worker_id: int):
        s = seed + worker_id
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)
    return _init_fn

def discover_seed_dirs(root: str, pattern: str) -> List[Path]:
    root_p = Path(root) # convert in a Path object the path of the root from string
    seed_dirs = [p for p in root_p.glob(pattern) if p.is_dir()] 
    # list of Path objects, order now is arbitrary and depends on filesystem traversal for example:  [Path(".../SEED_10"), Path(".../SEED_2"), Path(".../SEED_1")]

    def _key(p: Path):      # defined as an inner function because it is used only here
        m = SEED_DIR_RE.search(p.name)
        return int(m.group(1)) if m else 10**9
    
    """
    Assume: p = Path("/data/experiments/SEED_12")
    So p.name = "SEED_12"
    Then apply the regular expression: m = SEED_DIR_RE.search(p.name)
    and m will be either a match object if successfull or None
    In this case since p.name is "SEED_12" and the regex is "SEED_" followed by one or more digit, this will be a match. (While for example "SEED_first" will not match)
    Then m.group(1) for "SEED_12" will be "12" which is converted to integer.
    This means that the sorting will happend numerically and not lexicographically
    Notice the fallback for non-matching names, so if the directory name does not match the regular expression then the function returns (1,000,000,000)
    This is to push malformed directories to the end, and avoid crashes, because in these scenarios m will be None, and without that fallback this error raise: AttributeError: 'NoneType' object has no attribute 'group'
    """
    seed_dirs.sort(key=_key)
    return seed_dirs

# ==================================================
#     Dataset / Dataloader
# ==================================================
def find_mask_by_stem_png(masks_dir: Path, stem: str) -> Path:
    p = masks_dir / f"{stem}.png"
    if not p.exists():
        raise FileNotFoundError(f"Mask not found: {p}")
    return p

def build_loader(
    ds: Dataset,
    shuffle: bool,
    batch_size: int,
    cfg: CFG,
    run_seed: int,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(run_seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=shuffle,
        worker_init_fn=make_worker_init_fn(run_seed),
        generator=g,
    )

# ==================================================
#     Metrics / Evaluation
# ==================================================
def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-8) -> Dict[str, float]:
    tp = int(tp); fp = int(fp); fn = int(fn); tn = int(tn)
    p = tp + fn
    n = tn + fp
    pp = tp + fp
    pn = tn + fn
    total = p + n

    precision = tp / (pp + eps)
    recall    = tp / (p + eps)
    specificity = tn / (n + eps)
    fpr       = fp / (n + eps)
    fnr       = fn / (p + eps)
    iou       = tp / (tp + fp + fn + eps)
    dice      = (2 * tp) / (2 * tp + fp + fn + eps)
    acc       = (tp + tn) / (total + eps)
    bal_acc   = 0.5 * (recall + specificity)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "false_alarm_rate": float(fpr),
        "miss_rate": float(fnr),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
    }

def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))

def summarize_values(values):
    vals = [float(v) for v in values if _is_num(v)]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "sample_std": (stdev(vals) if len(vals) > 1 else 0.0),
        "population_std": (pstdev(vals) if len(vals) > 1 else 0.0),
        "min": min(vals),
        "max": max(vals),
    }

def aggregate_val_metrics_best(out_root: Path, seed_dirs: List[Path], num_classes: int) -> dict:
    """
    Looks for: out_root/SEED_xxx/val_metrics_best.json
    Aggregates:
      - global hard IoU (metrics['global']['iou'])
      - global soft IoU (metrics['global']['soft_iou'])
      - plus any other global scalar metrics you want
      - per-class metrics aggregated across seeds (skipping None / missing)
    """
    per_seed = []
    missing_files = []

    # collect per-seed dicts
    for sd in seed_dirs:
        seed_out = out_root / sd.name
        p = seed_out / "val_metrics_best.json"
        if not p.exists():
            missing_files.append(str(p))
            continue
        try:
            per_seed.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            missing_files.append(str(p))

    # ---- GLOBAL aggregation ----
    global_keys = [
        "iou", "soft_iou",
        "precision", "recall", "dice",
        "false_alarm_rate", "miss_rate",
        "accuracy", "balanced_accuracy",
    ]

    global_aggr = {}
    for k in global_keys:
        global_aggr[k] = summarize_values(
            [d.get("global", {}).get(k) for d in per_seed]
        )

    # ---- PER-CLASS aggregation ----
    per_class_aggr = {cid: {} for cid in range(num_classes)}
    per_class_keys = [
        "iou", "soft_iou",
        "precision", "recall", "dice",
        "false_alarm_rate", "miss_rate",
        "support_pixels",
    ]

    for cid in range(num_classes):
        cls_list = [d.get("per_class", {}).get(str(cid), d.get("per_class", {}).get(cid)) for d in per_seed]
        # cls_list entries are dicts or None
        for k in per_class_keys:
            per_class_aggr[cid][k] = summarize_values(
                [(c.get(k) if isinstance(c, dict) else None) for c in cls_list]
            )

    return {
        "num_seeds_found": len(per_seed),
        "missing_val_metrics_best": missing_files,
        "global": global_aggr,
        "per_class": per_class_aggr,
    }
# ==================================================
#    Dataset / Cutmix
# ==================================================
FSCAN_EXTS = {".fscan", ".h5"}

def list_fscans(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in FSCAN_EXTS]
    files.sort()
    return files

def build_defect_donor_indices_from_masks(train_masks_dir: str, donor_fscans: List[Path]) -> List[int]:
    mdir = Path(train_masks_dir)
    out = []
    for i, fp in enumerate(donor_fscans):
        mp = mdir / f"{fp.stem}.png"
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        # donor is "defect" if any pixel > 0
        if (m > 0).any():
            out.append(i)
    return out

def material_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]

class CleanFscanDataset(Dataset):
    def __init__(self, fscans: List[Path], cfg):
        self.fscans = fscans
        self.cfg = cfg
        if not self.fscans:
            raise FileNotFoundError("CleanFscanDataset: empty fscan list")

    def __len__(self): return len(self.fscans)

    def __getitem__(self, idx: int):
        p = self.fscans[idx]
        stem = p.stem

        x, _ = build_input_stack_from_fscan(
            fscan_path=p,
            streams=self.cfg.input_streams,
            fft_mode=self.cfg.fft_mode,
            lp_radius=self.cfg.fft_lp_radius,
            use_minmax_norm=self.cfg.use_minmax_norm,
            imagenet_norm=self.cfg.imagenet_norm,
            mean=self.cfg.mean,
            std=self.cfg.std,
        )  # [C,H,W]

        C, H, W = x.shape
        if (W, H) != (self.cfg.img_size, self.cfg.img_size):
            x = np.stack(
                [cv2.resize(x[c], (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_LINEAR)
                 for c in range(C)],
                axis=0
            ).astype(np.float32)

        return torch.from_numpy(x).float(), stem

class OnlineCutMixDatasetFscan(Dataset):
    """
    CutMix in channel-space.
    Donor provides (x_d, y_bin, y_idx, stem, vis_u8).
    Clean provides x_c.
    Paste: x_out[:, mask] = x_d[:, mask], keep labels from donor.
    """
    def __init__(
        self,
        donor_ds: Dataset,                 # train_ds_base (training split only)
        clean_fscans_dir: str,
        val_fscans_dir: str,               # used only to compute forbidden materials
        cfg,
        global_seed: int,
        train_masks: str,  
    ):
        self.cfg = cfg
        self.donor_ds = donor_ds
        self.global_seed = int(global_seed)
        self.train_masks = train_masks

        # Forbidden materials = materials present in validation split
        val_dir = Path(val_fscans_dir)
        val_fscans = []
        val_fscans += list(val_dir.glob("*.fscan"))
        val_fscans += list(val_dir.glob("*.h5"))
        self.val_materials = {material_from_stem(p.stem) for p in val_fscans}

        # Build clean pool and FILTER by validation materials
        clean_dir = Path(clean_fscans_dir)
        clean_all = list_fscans(clean_dir)
        clean_ok = [p for p in clean_all if material_from_stem(p.stem) not in self.val_materials]
        if not clean_ok:
            raise RuntimeError(
                f"No eligible clean fscans after filtering by val materials. "
                f"Clean total={len(clean_all)}, val_materials={len(self.val_materials)}"
            )

        self.clean_ds = CleanFscanDataset(clean_ok, cfg)

        K = int(cfg.cutmix_cleans_per_defect)
        if K <= 0:
            raise ValueError("cutmix_cleans_per_defect must be >= 1")

        Nclean = len(self.clean_ds)
        self.pairs: List[Tuple[int, int]] = []
   
       
        donor_fscans = donor_ds.fscans  # list[Path]
        donor_defect_idx = build_defect_donor_indices_from_masks(train_masks, donor_fscans)

        for di in donor_defect_idx:
            donor_stem = donor_fscans[di].stem
            k = min(K, Nclean)
            if k == Nclean:
                chosen = list(range(Nclean))
            else:
                rng = np.random.default_rng(per_donor_seed(self.global_seed, donor_stem))
                chosen = rng.choice(Nclean, size=k, replace=False).tolist()

            for ci in chosen:
                self.pairs.append((di, ci))

        if not self.pairs:
            raise RuntimeError("No CutMix pairs created (maybe all donors are defect-free?)")
        self.cutmix_stems = []
        for di, ci in self.pairs:
            donor_stem = self.donor_ds.fscans[di].stem
            clean_stem = self.clean_ds.fscans[ci].stem
            stem_out = f"{clean_stem}__cutmix__{donor_stem}"
            self.cutmix_stems.append(stem_out)


    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int):
        di, ci = self.pairs[idx]
        x_d, y_bin, y_idx, donor_stem, vis_u8 = self.donor_ds[di]
        x_c, clean_stem = self.clean_ds[ci]

        m = (y_bin > 0)  # [H,W] bool
        x_out = x_c.clone()
        x_out[:, m] = x_d[:, m]

        stem_out = f"{clean_stem}__cutmix__{donor_stem}"
        return x_out, y_bin, y_idx, stem_out, vis_u8
    
class SegDatasetFscanIndexed(Dataset):
    def __init__(self, fscans_dir: str, masks_dir: str, cfg, dump_dir: Optional[Path] = None):
        self.cfg = cfg
        self.fscans_dir = Path(fscans_dir)
        self.masks_dir = Path(masks_dir)
        self.fscans = list_fscans(self.fscans_dir)
        if not self.fscans:
            raise FileNotFoundError(f"No fscans found in: {self.fscans_dir}")

        self.dump_dir = dump_dir
        self._dumped = 0

        # overlay stream must be present in input_streams
        if cfg.overlay_stream not in cfg.input_streams:
            raise ValueError(
                f"overlay_stream='{cfg.overlay_stream}' not in input_streams={cfg.input_streams}"
            )
        self._vis_idx = list(cfg.input_streams).index(cfg.overlay_stream)

    def __len__(self): return len(self.fscans)

    def __getitem__(self, idx: int):
        fscan_path = self.fscans[idx]
        stem = fscan_path.stem

        mask_path = find_mask_by_stem_png(self.masks_dir, stem)

        x, x_unnormalized_float32 = build_input_stack_from_fscan(
            fscan_path=fscan_path,
            streams=self.cfg.input_streams,
            fft_mode=self.cfg.fft_mode,
            lp_radius=self.cfg.fft_lp_radius,
            use_minmax_norm=self.cfg.use_minmax_norm,
            imagenet_norm=self.cfg.imagenet_norm,
            mean=self.cfg.mean,
            std=self.cfg.std,
        )  # [C,H,W] float32

        # resize x
        C, H, W = x.shape
        if (W, H) != (self.cfg.img_size, self.cfg.img_size):
            x = np.stack(
                [cv2.resize(x[c], (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_LINEAR) for c in range(C)],
                axis=0
            ).astype(np.float32)

       
        vis_ch = x_unnormalized_float32[self._vis_idx]  # should be [H,W]

        # if not already model size, resize it
        if vis_ch.shape != (self.cfg.img_size, self.cfg.img_size):
            vis_ch = cv2.resize(
                vis_ch,
                (self.cfg.img_size, self.cfg.img_size),
                interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)

        vis_u8 = to_u8_from_01(vis_ch)  # [H,W] uint8 
        # indexed mask
        m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        if m.ndim != 2:
            raise RuntimeError(f"Wrong number of channel for input mask: {mask_path}")
        # resize mask
        if (m.shape[1], m.shape[0]) != (self.cfg.img_size, self.cfg.img_size):
            m = cv2.resize(m, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_NEAREST)

        # ---------------------------------------------------
        # FAIL-FAST: ensure indexed labels are within range
        # (this does NOT require max==num_defect_classes,
        # it only checks max does not exceed it)
        # ---------------------------------------------------
        minv = int(m.min()) if m.size else 0
        maxv = int(m.max()) if m.size else 0
        if minv < 0:
            raise ValueError(f"Mask has negative labels (min={minv}) for: {mask_path}")
        if maxv > self.cfg.num_defect_classes:
            raise ValueError(
                f"Mask label out of range: max={maxv} but cfg.num_defect_classes={self.cfg.num_defect_classes} "
                f"for: {mask_path}"
            )

        y_idx = m.astype(np.int64)               # 0..num_classes, indexed mask
        y_bin = (y_idx > 0).astype(np.int64)     # {0,1} binarized mask

        return (
            torch.from_numpy(x).float(),                 # x
            torch.from_numpy(y_bin).long(),              # y_bin
            torch.from_numpy(y_idx).long(),              # y_idx
            stem,
            vis_u8,
        )
    
# ==================================================
#     Visualization / Utilities
# ==================================================
def save_cfg(cfg, out_dir: Path, filename: str = "config.json"):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    with open(out_dir / filename, "w") as f:
        json.dump(cfg_dict, f, indent=2)

def print_seed_recap(cfg: CFG, run_seed: int, seed_dir: Path,
                     n_train_base: int, n_val: int,
                     n_cutmix: int = 0) -> None:
    in_ch = len(cfg.input_streams)

    print("\n" + "=" * 88)
    print(f"[SEED] {seed_dir.name} | seed={run_seed}")
    print(f"[SPLIT] train_base={n_train_base} | valid={n_val}")

    if cfg.cutmix_enable:
        print(f"[CUTMIX] enabled=True | clean_dir='{cfg.cutmix_clean_fscans_dir}' | "
              f"K={cfg.cutmix_cleans_per_defect} | cutmix_samples={n_cutmix} | "
              f"epoch_len={n_train_base + n_cutmix}")
    else:
        print("[CUTMIX] enabled=False")

    print(f"[INPUT] streams={cfg.input_streams} | in_channels={in_ch}")
    print(f"[FFT]   fft_mode={cfg.fft_mode} | lp_radius={cfg.fft_lp_radius}")
    print(f"[NORM]  minmax={cfg.use_minmax_norm} | imagenet_norm={cfg.imagenet_norm}")
    print(f"[MODEL] hf_ckpt='{cfg.hf_ckpt}' | drop_path={cfg.drop_path_rate} | inproj_init={cfg.inproj_init}")
    print(f"[TRAIN] epochs={cfg.epochs} | bs={cfg.batch_size} | lr={cfg.lr} | wd={cfg.weight_decay} | amp={cfg.use_amp}")
    print(f"[EVAL]  thr={cfg.sel_thr} | num_defect_classes={cfg.num_defect_classes}")
    print("=" * 88 + "\n")

def overlay_mask_rgb(base: np.ndarray, mask01: np.ndarray, color_rgb=(0,255,0), alpha: float = 0.45) -> np.ndarray:
    """
    base:   [H,W] uint8 OR [H,W,3] uint8
    mask01: [H,W] or [H,W,1] (0/1 or bool)
    returns [H,W,3] uint8 (RGB)
    """
    if base.ndim == 2:
        base = np.repeat(base[:, :, None], 3, axis=2)
    elif base.ndim == 3 and base.shape[2] == 1:
        base = np.repeat(base, 3, axis=2)          # [H,W,1] -> [H,W,3]
    elif base.ndim == 3 and base.shape[2] == 3:
        base = base.copy()
    else:
        raise ValueError(f"overlay_mask_rgb: base must be [H,W] or [H,W,3], got {base.shape}")

    if mask01.ndim == 3:
        mask01 = mask01[..., 0]
    m = mask01.astype(bool)  # [H,W]

    out_f = base.astype(np.float32)
    color = np.array(color_rgb, dtype=np.float32)

    out_f[m] = (1.0 - alpha) * out_f[m] + alpha * color
    return np.clip(out_f, 0, 255).astype(np.uint8)

def to_u8_from_01(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0).astype(np.uint8)

def denormalize_imagenet(x: np.ndarray,
                         mean: Tuple[float, float, float],
                         std: Tuple[float, float, float]) -> np.ndarray:
    """
    x: [C,H,W] numpy float32, ImageNet-normalized
    returns: [C,H,W] float32 in original scale
    """
    m = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    s = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return x * s + m

def resolve_eval_ckpt(cfg: CFG, seed_out: Path) -> Path:
    """
    Eval policy:
    - if cfg.ckpt_path is provided: use that checkpoint for ALL seeds
    - else: use per-seed best.pt in the seed output folder
    """
    if cfg.ckpt_path:
        return Path(cfg.ckpt_path)
    return seed_out / "best.pt"

def dump_tensor_sample(out_dir: Path, stem: str, x_t: torch.Tensor, cfg: CFG):
    """
    x_t: tensor [C,H,W] coming directly from DataLoader
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    x = x_t.detach().cpu().numpy().astype(np.float32)  # [C,H,W]
    C, H, W = x.shape

    # --------------------------------------------------
    # Undo ImageNet normalization ONLY for visualization
    # --------------------------------------------------
    if cfg.imagenet_norm and C == 3:
        mean = np.asarray(cfg.mean, dtype=np.float32).reshape(3, 1, 1)
        std  = np.asarray(cfg.std,  dtype=np.float32).reshape(3, 1, 1)
        x = x * std + mean   # inverse normalization

    # --------------------------------------------------
    # Now assume values are in [0,1] (or approximately)
    # Just clip and scale to 0–255
    # --------------------------------------------------

    def to_u8_from_01(x01: np.ndarray) -> np.ndarray:
        x01 = np.clip(x01, 0.0, 1.0)
        return (x01 * 255.0).astype(np.uint8)

    # Case 1 — exactly 3 channels → RGB image
    if C == 3:
        rgb = np.stack([to_u8_from_01(x[c]) for c in range(3)], axis=-1)
        cv2.imwrite(str(out_dir / f"{stem}.png"), rgb)

    # Case 2 — more than 3 channels → save each channel separately
    else:
        for c in range(C):
            ch = to_u8_from_01(x[c])
            cv2.imwrite(str(out_dir / f"{stem}_ch{c:02d}.png"), ch)

@torch.no_grad()
def dump_first_n_from_loader(loader, out_dir: Path, cfg: CFG):

    saved = 0
    n = cfg.dump_inputs_max
    for x, y_bin, y_idx, stems, vis_u8 in loader:
        B = x.size(0)

        for b in range(B):
            if saved >= n:
                return

            dump_tensor_sample(out_dir, stems[b], x[b], cfg)
            saved += 1

def channel_to_vis_rgb_u8(ch2d: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    x = ch2d.astype(np.float32)
    a = float(np.percentile(x, lo))
    b = float(np.percentile(x, hi))
    if b <= a + 1e-8:
        x01 = np.zeros_like(x, dtype=np.float32)
    else:
        x01 = np.clip((x - a) / (b - a), 0.0, 1.0)
    u8 = (x01 * 255.0).astype(np.uint8)
    return np.repeat(u8[:, :, None], 3, axis=2)

def save_plots(history: Dict[str, List[float]], out_dir: Path) -> None:
    """
    Creates two simple plots:
      - loss curves
      - val soft IoU curve
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # loss plot
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png")
    plt.close()

    # iou plot
    plt.figure()
    plt.plot(epochs, history["val_soft_iou_def"], label="val_soft_iou_def")
    plt.xlabel("epoch")
    plt.ylabel("soft IoU (defect)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_iou_curve.png")
    plt.close()

def seed_is_completed(seed_out: Path) -> bool:
    return (seed_out / "val_metrics_best.json").is_file()

def filter_incomplete_seeds(seed_dirs: List[Path], out_root: Path) -> List[Path]:
    todo = []
    done = []
    for sd in seed_dirs:
        seed_out = out_root / sd.name
        if seed_is_completed(seed_out):
            done.append(sd)
        else:
            todo.append(sd)

    if done:
        print(f"[RESUME] completed seeds ({len(done)}): {[s.name for s in done]}")
    if todo:
        print(f"[RESUME] remaining seeds ({len(todo)}): {[s.name for s in todo]}")
    return todo
# ==================================================
#    Train / Validation
# ==================================================
def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: Optional[GradScaler], epoch: int, best_metric: float, cfg: CFG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "best_metric": float(best_metric),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": asdict(cfg),
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, str(path))

def load_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scaler: Optional[GradScaler] = None,
                    map_location: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise ValueError(f"Checkpoint must be a dict with key 'model': {path}")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"[CKPT] Loaded model. Missing={len(missing)} Unexpected={len(unexpected)}")
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("[CKPT] Loaded optimizer state.")
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
        print("[CKPT] Loaded scaler state.")
    return ckpt

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    use_amp
) -> Tuple[float, float]:
    model.train()
    loss_sum, n_images = 0.0, 0
    cuda_sync()
    t0 = time.perf_counter()

    for x, y_bin, *_ in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y_bin = y_bin.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            out = model(pixel_values=x)
            logits = out.logits
            if logits.shape[-2:] != y_bin.shape[-2:]:
                logits = F.interpolate(logits, size=y_bin.shape[-2:], mode="bilinear", align_corners=False)
            # logits: [B,1,H,W], y: [B,H,W]
            t = y_bin.unsqueeze(1).float()     # [B,1,H,W]
            loss = criterion(logits, t)


        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.item()) * x.size(0)
        n_images += x.size(0)
    cuda_sync()
    elapsed = time.perf_counter() - t0

    return loss_sum / max(n_images, 1), elapsed

@torch.no_grad()
def forward_logits(model, x, target_hw, use_amp: bool):
    with autocast(device_type="cuda", enabled=use_amp):
        out = model(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != target_hw:
            logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
    return logits

def probs_and_pred_from_logits(logits, thr: float):
    probs = torch.sigmoid(logits[:, 0].float())   # [B,H,W]
    pred  = (probs > thr)                         # bool
    return probs, pred

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    use_amp: bool,
    eps: float = 1e-8
) -> Tuple[float, float, float]:
    model.eval()

    cuda_sync()
    t0 = time.perf_counter()
    loss_sum = 0.0
    n_images = 0
    inter_total = 0.0
    union_total = 0.0

    for x, y_bin, *_ in loader:
        x = x.to(device, non_blocking=True)
        y_bin = y_bin.to(device, non_blocking=True)  # [B,H,W] long {0,1}

        logits = forward_logits(model, x, y_bin.shape[-2:], use_amp)
        t = y_bin.unsqueeze(1).float()
        loss = criterion(logits, t)
        probs = torch.sigmoid(logits[:, 0].float())

        y_f = y_bin.float()
        inter = (probs * y_f).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + y_f.sum(dim=(1, 2)) - inter

        loss_sum += float(loss.item()) * x.size(0)
        n_images += x.size(0)
        inter_total += float(inter.sum().item())
        union_total += float(union.sum().item())
    cuda_sync()
    elapsed = time.perf_counter() - t0
    if n_images <= 0: 
        raise RuntimeError("[VALIDATE_ONE_EPOCH] Expected at least one validation image, got zero.")
    val_loss = loss_sum / max(n_images, 1)
    val_soft_iou_global = (inter_total + eps) / (union_total + eps)

    if not np.isfinite(val_soft_iou_global):
        raise RuntimeError(
            f"val_soft_iou_def became non-finite: inter_total={inter_total}, union_total={union_total}"
        )

    return val_loss, float(val_soft_iou_global), elapsed

@torch.no_grad()
def evaluate_split(
    model,
    loader,
    device,
    use_amp: bool,
    thr: float,
    num_classes: int,
    eps: float = 1e-8,
    out_vis_dir: Optional[Path] = None,
    vis_alpha: float = 0.45,
    max_vis: int = 200,
) -> Dict[str, object]:
    """
    Same as your evaluate_split BUT fixes per-class confusion to be one-vs-rest:
      positives = gt_k
      negatives = ~gt_k  (background + other defects)
    """
    model.eval()

    g_tp = g_fp = g_fn = g_tn = 0

    per_cls_counts = {
        cid: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "support_pixels": 0}
        for cid in range(num_classes)
    }

    soft_inter_total = 0.0
    soft_union_total = 0.0
    per_cls_soft_inter = {cid: 0.0 for cid in range(num_classes)}
    per_cls_soft_union = {cid: 0.0 for cid in range(num_classes)}

    saved = 0
    if out_vis_dir is not None:
        out_vis_dir.mkdir(parents=True, exist_ok=True)

    for x, y_bin, y_idx, stems, vis_u8 in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y_bin = y_bin.to(device, non_blocking=True)
        y_idx = y_idx.to(device, non_blocking=True)

        logits = forward_logits(model, x, y_bin.shape[-2:], use_amp)
        probs, pred = probs_and_pred_from_logits(logits, thr=thr)

        gt = (y_bin == 1)

        gt_f = gt.float()
        inter = (probs * gt_f).sum()
        union = probs.sum() + gt_f.sum() - inter
        soft_inter_total += float(inter.item())
        soft_union_total += float(union.item())

        # global confusion
        g_tp += int((pred & gt).sum().item())
        g_fp += int((pred & ~gt).sum().item())
        g_fn += int((~pred & gt).sum().item())
        g_tn += int((~pred & ~gt).sum().item())

        # per-class one-vs-rest
        for cid in range(num_classes):
            gt_k = (y_idx == (cid + 1))
            neg_k = ~gt_k

            tp = int((pred & gt_k).sum().item())
            fn = int((~pred & gt_k).sum().item())
            fp = int((pred & neg_k).sum().item())
            tn = int((~pred & neg_k).sum().item())

            per_cls_counts[cid]["tp"] += tp
            per_cls_counts[cid]["fp"] += fp
            per_cls_counts[cid]["fn"] += fn
            per_cls_counts[cid]["tn"] += tn
            per_cls_counts[cid]["support_pixels"] += int(gt_k.sum().item())

            # soft IoU for class (only if present)
            if gt_k.any():
                gk = gt_k.float()
                inter_k = (probs * gk).sum()
                union_k = probs.sum() + gk.sum() - inter_k
                per_cls_soft_inter[cid] += float(inter_k.item())
                per_cls_soft_union[cid] += float(union_k.item())

        # overlays
        if out_vis_dir is not None and saved < max_vis:
            # vis_u8 comes from dataset: [B,H,W] uint8 (single channel)
            vis_np = vis_u8.detach().cpu().numpy() if torch.is_tensor(vis_u8) else np.asarray(vis_u8)

            pred_np  = pred.detach().cpu().numpy().astype(np.uint8)     # [B,H,W] 0/1
            ybin_np  = y_bin.detach().cpu().numpy().astype(np.uint8)    # [B,H,W] 0/1

            B = pred_np.shape[0]
            for b in range(B):
                if saved >= max_vis:
                    break

                stem = stems[b]
                base_gray = vis_np[b]  # [H,W] uint8

                gt01 = (ybin_np[b] > 0).astype(np.uint8)   # [H,W]
                pr01 = (pred_np[b] > 0).astype(np.uint8)   # [H,W]

                # Convert gray->RGB inside overlay_mask_rgb and colorize
                left  = overlay_mask_rgb(base_gray, gt01, color_rgb=(0, 255, 0), alpha=vis_alpha)   # GT green
                right = overlay_mask_rgb(base_gray, pr01, color_rgb=(255, 0, 0), alpha=vis_alpha)   # Pred red

                side = np.concatenate([left, right], axis=1)  # [H,2W,3] RGB
                cv2.imwrite(str(out_vis_dir / f"{stem}.png"), side[:, :, ::-1])  # RGB->BGR for OpenCV
                saved += 1

    global_metrics = metrics_from_confusion(g_tp, g_fp, g_fn, g_tn, eps=eps)
    global_metrics["soft_iou"] = (soft_inter_total + eps) / (soft_union_total + eps)

    per_class_metrics = {}
    present = []
    for cid, c in per_cls_counts.items():
        m = metrics_from_confusion(c["tp"], c["fp"], c["fn"], c["tn"], eps=eps)
        m["support_pixels"] = int(c["support_pixels"])
        if c["support_pixels"] > 0:
            m["soft_iou"] = (per_cls_soft_inter[cid] + eps) / (per_cls_soft_union[cid] + eps)
            present.append(cid)
        else:
            m["soft_iou"] = None
        per_class_metrics[cid] = m

    rankings = {}
    if present:
        rankings["most_missed_class"] = max(present, key=lambda cid: per_class_metrics[cid]["miss_rate"])
        rankings["best_iou_class"] = max(present, key=lambda cid: per_class_metrics[cid]["iou"])
        rankings["worst_iou_class"] = min(present, key=lambda cid: per_class_metrics[cid]["iou"])
        rankings["highest_false_alarm_rate_class"] = max(present, key=lambda cid: per_class_metrics[cid]["false_alarm_rate"])

    return {"global": global_metrics, "per_class": per_class_metrics, "rankings": rankings}
# ==================================================
#                       MAIN
# ==================================================
def main():
    cfg = build_cfg_from_args()
    cfg.device = resolve_device(cfg.device)
    validate_cfg(cfg)

    if cfg.device.startswith("cpu"):
        print("[WARN] Running on CPU. This script was developed primarily for GPU execution and may be very slow.")
    out_root = Path(cfg.out_dir) / cfg.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    seed_dirs = discover_seed_dirs(cfg.seeds_root, cfg.seed_glob)
    save_cfg(cfg, out_root)

    # -----------------------
    # EVAL MODE
    # -----------------------
    if cfg.mode.lower() == "eval":
        out_root = Path(cfg.out_dir) / cfg.run_name
        for seed_dir in seed_dirs:
            seed_out = out_root / seed_dir.name
            ckpt_path = resolve_eval_ckpt(cfg, seed_out)

            if not ckpt_path.exists():
                raise FileNotFoundError(f"[EVAL] checkpoint not found: {ckpt_path}")

            m = SEED_DIR_RE.search(seed_dir.name)
            if not m:
                continue
            run_seed = int(m.group(1))
            seed_everything(run_seed, deterministic=cfg.deterministic)

            train_fscans = seed_dir / "train" / cfg.fscan_subdir
            train_masks  = seed_dir / "train" / cfg.masks_subdir
            val_fscans   = seed_dir / "valid" / cfg.fscan_subdir
            val_masks    = seed_dir / "valid" / cfg.masks_subdir

            seed_out = out_root / seed_dir.name

            # build val loader for THIS seed
            val_ds = SegDatasetFscanIndexed(str(val_fscans), str(val_masks), cfg, dump_dir=None)
            val_loader = build_loader(val_ds, shuffle=False, batch_size=cfg.batch_size, cfg=cfg, run_seed=run_seed)

            # build model (same config used in training)
            model = build_model(cfg).to(cfg.device)
            load_checkpoint(str(ckpt_path), model, optimizer=None, scaler=None, map_location=cfg.device)

            vis_dir = seed_out / "eval_overlays_best"
            metrics = evaluate_split(
                model, val_loader, device=cfg.device, use_amp=cfg.use_amp,
                thr=cfg.sel_thr, num_classes=cfg.num_defect_classes,
                out_vis_dir=vis_dir, vis_alpha=cfg.vis_alpha, max_vis=cfg.max_vis,
            )
            (seed_out / "val_metrics_best.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print("[OK] Eval completed.")
        summary = aggregate_val_metrics_best(out_root, seed_dirs, num_classes=cfg.num_defect_classes)
        (out_root / "val_metrics_best_aggregate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return

    # -----------------------
    # TRAIN MODE
    # -----------------------
    
    if cfg.mode.lower() == "train":
            # Skip already-finished seeds
            seed_dirs_all = seed_dirs
            seed_dirs_todo = filter_incomplete_seeds(seed_dirs_all, out_root)
            if not seed_dirs_todo:
                print("[OK] All seeds are already completed (val_metrics_best.json exists). Nothing to do.")
                return

            for seed_dir in seed_dirs_todo:

                m = SEED_DIR_RE.search(seed_dir.name)
                if not m:
                    print(f"[WARN] Skip wrong name for seed dir: {seed_dir}")
                    continue
                run_seed = int(m.group(1))
                seed_everything(run_seed, deterministic=cfg.deterministic)
                train_fscans = seed_dir / "train" / cfg.fscan_subdir
                train_masks  = seed_dir / "train" / cfg.masks_subdir
                val_fscans   = seed_dir / "valid" / cfg.fscan_subdir
                val_masks    = seed_dir / "valid" / cfg.masks_subdir
                seed_out = out_root / seed_dir.name
                seed_out.mkdir(parents=True, exist_ok=True)

                # base loaders (donors must be training split only)
                train_ds_base = SegDatasetFscanIndexed(str(train_fscans), str(train_masks), cfg, dump_dir=None)
                train_loader  = build_loader(train_ds_base, shuffle=True, batch_size=cfg.batch_size, cfg=cfg, run_seed=run_seed)

                val_ds        = SegDatasetFscanIndexed(str(val_fscans), str(val_masks), cfg, dump_dir=None)
                val_loader    = build_loader(val_ds, shuffle=False, batch_size=cfg.batch_size, cfg=cfg, run_seed=run_seed)

                # optional CutMix
                cutmix_ds = None
                n_cutmix = 0

                if cfg.cutmix_enable:
                    cutmix_ds = OnlineCutMixDatasetFscan( 
                        donor_ds=train_ds_base,
                        clean_fscans_dir=cfg.cutmix_clean_fscans_dir,
                        val_fscans_dir=str(val_fscans),
                        cfg=cfg,
                        global_seed=run_seed,
                        train_masks=str(train_masks)
                    )
                    n_cutmix = len(cutmix_ds)
                    (seed_out / "cutmix_stems.json").write_text(json.dumps(cutmix_ds.cutmix_stems, indent=2), encoding="utf-8")
                    train_ds = ConcatDataset([train_ds_base, cutmix_ds])
                else:
                    train_ds = train_ds_base

                print_seed_recap(cfg, run_seed, seed_dir, n_train_base=len(train_ds_base), n_val=len(val_ds), n_cutmix=n_cutmix)


                # final train loader from FINAL dataset
                train_loader = build_loader(train_ds, shuffle=True, batch_size=cfg.batch_size, cfg=cfg, run_seed=run_seed)
                
                # dump AFTER final loader exists (dumps cutmix outputs if enabled)
                if cfg.dump_inputs:
                    dump_dir = seed_out / cfg.dump_inputs_dirname
                    dump_first_n_from_loader(train_loader, Path(dump_dir), cfg)
                    print(f"[DUMP] Saved {cfg.dump_inputs_max} samples to {dump_dir}")

                # model
                model = build_model(cfg).to(cfg.device)

                optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                scaler = GradScaler(enabled=cfg.use_amp)
                criterion = FocalTverskyLoss(
                    alpha_focal=cfg.alpha_focal,
                    gamma_focal=cfg.gamma_focal,
                    wf=cfg.wf,
                    alpha_tv=cfg.alpha_tv,
                    beta_tv=cfg.beta_tv,
                )

                # resume training (optional)
                start_epoch = 1
                best_soft_iou = -1.0
                resume_pt = getattr(cfg, "resume_pt", "")
                if resume_pt:
                    print(f"[RESUME] from {resume_pt}")
                    ckpt = load_checkpoint(resume_pt, model, optimizer=optimizer, scaler=scaler, map_location=cfg.device)
                    start_epoch = int(ckpt.get("epoch", 0)) + 1
                    best_soft_iou = float(ckpt.get("best_metric", -1.0))

                history = {
                    "train_loss": [],
                    "val_loss": [],
                    "val_soft_iou_def": [],
                    "train_time_sec": [],
                    "val_time_sec": []
                }

                best_path = seed_out / "best.pt"
                cuda_sync()
                seed_t0 = time.perf_counter()

                for epoch in range(start_epoch, cfg.epochs + 1):
                    train_loss, train_time = train_one_epoch(model, train_loader, optimizer, scaler, criterion, cfg.device, cfg.use_amp)
                    val_loss, val_soft_iou, val_time  = validate_one_epoch(model, val_loader, criterion, cfg.device, cfg.use_amp)

                    history["train_loss"].append(float(train_loss))
                    history["val_loss"].append(float(val_loss))
                    history["val_soft_iou_def"].append(float(val_soft_iou))
                    
                    history["train_time_sec"].append(float(train_time))
                    history["val_time_sec"].append(float(val_time))

                    print(
                        f"[{seed_dir.name}] epoch={epoch:03d} "
                        f"train_loss={train_loss:.5f} "
                        f"val_loss={val_loss:.5f} "
                        f"val_soft_iou={val_soft_iou:.5f} "
                        f"train_time={train_time:.1f}s "
                        f"val_time={val_time:.1f}s"
                    )


                    # periodic checkpoint
                    if cfg.ckpt_every and (epoch % cfg.ckpt_every == 0):
                        save_checkpoint(seed_out / f"epoch_{epoch:03d}.pt", model, optimizer, scaler, epoch, best_soft_iou, cfg)

                    # best checkpoint
                    if val_soft_iou > best_soft_iou:
                        best_soft_iou = float(val_soft_iou)
                        save_checkpoint(best_path, model, optimizer, scaler, epoch, best_soft_iou, cfg)
                        print(f"[BEST] new best soft IoU={best_soft_iou:.6f} at epoch {epoch}")

                    # save history each epoch
                    (seed_out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

                cuda_sync()
                total_train_time = time.perf_counter() - seed_t0

                history["total_train_time_sec"] = float(total_train_time)
                # plots (optional)
                try:
                    save_plots(history, seed_out)
                except Exception as e:
                    print(f"[WARN] save_plots failed: {e}")

                # FINAL: evaluate best checkpoint with overlays on validation
                print("[FINAL] Loading best checkpoint for final eval + overlays...")
                load_checkpoint(str(best_path), model, optimizer=None, scaler=None, map_location=cfg.device)
                cuda_sync()
                t0 = time.perf_counter()
                final_vis = seed_out / "val_overlays_best"
                final_metrics = evaluate_split(
                    model, val_loader, device=cfg.device, use_amp=cfg.use_amp,
                    thr=cfg.sel_thr, num_classes=cfg.num_defect_classes,
                    out_vis_dir=final_vis, vis_alpha=cfg.vis_alpha, max_vis=cfg.max_vis
                )
                final_eval_time = time.perf_counter() - t0
                final_metrics["evaluation_time_sec"] = float(final_eval_time)

                (seed_out / "val_metrics_best.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
                print(f"[OK] Final metrics saved: {seed_out/'val_metrics_best.json'}")

            print("\n[OK] Training completed for all seeds.")
            summary = aggregate_val_metrics_best(out_root, seed_dirs_all, num_classes=cfg.num_defect_classes)
            (out_root / "val_metrics_best_aggregate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[OK] Aggregated metrics saved to: {out_root/'val_metrics_best_aggregate.json'}")



if __name__ == "__main__":
    main()
