from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from nn.functional import (
    split_mat_files,
    plot_loss_curves,
    seed_everything,
    count_trainable_params,
    load_mat_as_tensor,
)


# =========================================================
# Model
# =========================================================
class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        )
        if in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
            )
        else:
            self.short = nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + self.short(x))


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        hidden = max(ch // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, kernel_size=1, bias=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(hidden, ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvNormAct(in_ch + skip_ch, out_ch, 3, 1, 1)
        self.res = ResBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.res(x)
        return x


class CT2PressureResUNet(nn.Module):
    """
    Input : CT crop        (B,1,128,128)
    Output: Pressure map   (B,1,512,512)
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_ch: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, c1, 3, 1, 1),
            ResBlock(c1, c1),
        )
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(c1, c2), ResBlock(c2, c2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(c2, c3), ResBlock(c3, c3))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), ResBlock(c3, c4), ResBlock(c4, c4))
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(c4, c5),
            ResBlock(c5, c5),
            CBAM(c5),
        )

        self.up4 = UpBlock(c5, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.head_256 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvNormAct(c1, c1, 3, 1, 1),
            ResBlock(c1, c1),
        )
        self.head_512 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvNormAct(c1, c1 // 2, 3, 1, 1),
            ResBlock(c1 // 2, c1 // 2),
        )
        self.out_conv = nn.Conv2d(c1 // 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got {tuple(x.shape)}")
        s1 = self.stem(x)        # 128
        s2 = self.enc2(s1)       # 64
        s3 = self.enc3(s2)       # 32
        s4 = self.enc4(s3)       # 16
        b = self.bottleneck(s4)  # 8

        x = self.up4(b, s4)      # 16
        x = self.up3(x, s3)      # 32
        x = self.up2(x, s2)      # 64
        x = self.up1(x, s1)      # 128
        x = self.head_256(x)     # 256
        x = self.head_512(x)     # 512
        x = self.out_conv(x)
        return x


# =========================================================
# Dataset / masked losses
# =========================================================
def _ensure_hw_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        if x.shape[0] == 1:
            return x[0]
        if x.shape[-1] == 1:
            return x[..., 0]
    raise ValueError(f"{name} expected 2D or singleton-channel tensor, got shape {tuple(x.shape)}")


def _pad_pressure_to_square(target_hw: torch.Tensor, pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, int]:
    h, w = target_hw.shape
    if h != 512:
        raise ValueError(f"Pressure map height must be 512, got {h}")
    if w > 512:
        raise ValueError(f"Pressure map width must be <= 512, got {w}")

    padded = torch.full((512, 512), float(pad_value), dtype=target_hw.dtype)
    padded[:, :w] = target_hw

    valid_mask = torch.zeros((512, 512), dtype=torch.float32)
    valid_mask[:, :w] = 1.0
    return padded, valid_mask, int(w)


class VxmCTPressurePairDataset(Dataset):
    """
    Match CT file by stem and right-pad pressure map from (512, n) to (512, 512).
    """
    def __init__(
        self,
        ct_dir: Path,
        target_paths: Sequence[Path],
        ct_key: str = "ct_crop",
        target_key: str = "p_max_ROI",
        normalize_ct: bool = False,
        normalize_target: bool = False,
        pad_value: float = 0.0,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.ct_dir = Path(ct_dir)
        self.ct_key = ct_key
        self.target_key = target_key
        self.normalize_ct = bool(normalize_ct)
        self.normalize_target = bool(normalize_target)
        self.pad_value = float(pad_value)

        ct_by_stem = {p.stem: p for p in self.ct_dir.rglob("*.mat")}
        self.samples: List[Tuple[Path, Path]] = []
        for tp in target_paths:
            tp = Path(tp)
            cp = ct_by_stem.get(tp.stem, None)
            if cp is not None:
                self.samples.append((cp, tp))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No matched CT/target pairs found under ct_dir={self.ct_dir} for {len(target_paths)} targets."
            )

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ct_path, target_path = self.samples[idx]

        ct = load_mat_as_tensor(ct_path, self.ct_key,
                                normalize=self.normalize_ct).float()
        target = load_mat_as_tensor(target_path, self.target_key,
                                    normalize=self.normalize_target).float()

        if not torch.isfinite(ct).all():
            raise ValueError(f"CT has NaN/Inf: {ct_path}")
        if not torch.isfinite(target).all():
            raise ValueError(f"Target has NaN/Inf before pad: {target_path}")

        ct = _ensure_hw_tensor(ct, "ct")
        target = _ensure_hw_tensor(target, "target")

        target_pad, valid_mask, orig_w = _pad_pressure_to_square(target,
                                                                 pad_value=self.pad_value)

        if not torch.isfinite(target_pad).all():
            raise ValueError(f"Target has NaN/Inf after pad: {target_path}")
        if not torch.isfinite(valid_mask).all():
            raise ValueError(f"Mask has NaN/Inf: {target_path}")

        return {
            "ct": ct.unsqueeze(0),
            "target": target_pad.unsqueeze(0),
            "valid_mask": valid_mask.unsqueeze(0),
            "orig_w": torch.tensor(orig_w, dtype=torch.long),
            "stem": target_path.stem,
        }


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(eps)


class MaskedMSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean((pred - target) ** 2, mask)


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = float(beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        beta = self.beta
        if beta < 1e-8:
            loss = diff
        else:
            loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        return masked_mean(loss, mask)


class MaskedSpatialGradientLoss(nn.Module):
    """L1 gradient consistency on valid region only."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
        dx_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]

        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
        dy_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        loss_x = masked_mean(torch.abs(dx_pred - dx_target), dx_mask)
        loss_y = masked_mean(torch.abs(dy_pred - dy_target), dy_mask)
        return 0.5 * (loss_x + loss_y)


# =========================================================
# Visualization
# =========================================================
@torch.no_grad()
def infer_5cts_and_plot_ct2pressure(
    model: nn.Module,
    target_paths: List[Path],
    ct_dir: Path,
    device: str,
    out_png: Path,
    ct_key: str = "ct_crop",
    target_key: str = "p_max_ROI",
    normalize_ct: bool = False,
    normalize_target: bool = False,
    pad_value: float = 0.0,
):
    import matplotlib.pyplot as plt

    model.eval()
    n = len(target_paths)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    ct_by_stem = {p.stem: p for p in ct_dir.rglob("*.mat")}

    for i, tp in enumerate(target_paths):
        if tp.stem not in ct_by_stem:
            raise FileNotFoundError(f"No CT file matched target stem: {tp.stem}")
        cp = ct_by_stem[tp.stem]

        ct = load_mat_as_tensor(cp, ct_key, normalize=normalize_ct).float()
        target = load_mat_as_tensor(tp, target_key, normalize=normalize_target).float()

        ct_hw = _ensure_hw_tensor(ct, "ct")
        target_hw = _ensure_hw_tensor(target, "target")
        target_pad, _, orig_w = _pad_pressure_to_square(target_hw, pad_value=pad_value)

        pred_pad = model(ct_hw.unsqueeze(0).unsqueeze(0).to(device)).detach().cpu()[0, 0]

        target_show = target_pad[:, :orig_w].numpy()
        pred_show = pred_pad[:, :orig_w].numpy()
        ct_show = ct_hw.numpy()

        target_show = (target_show - target_show.min()) / (target_show.max() - target_show.min() + 1e-8)
        pred_show = (pred_show - pred_show.min()) / (pred_show.max() - pred_show.min() + 1e-8)

        err_show = np.abs(pred_show - target_show)

        vmin, vmax = 0.0, 1.0

        ax = axes[i]
        ax[0].imshow(ct_show, cmap="gray")
        ax[0].set_title(f"CT\n{tp.stem}")

        ax[1].imshow(target_show, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
        ax[1].set_title(f"Target Pressure\n(512 x {orig_w})")

        ax[2].imshow(pred_show, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
        ax[2].set_title("Pred Pressure")

        ax[3].imshow(err_show, cmap="magma", aspect="auto")
        ax[3].set_title("|Pred-Target|")

        for a in ax:
            a.axis("off")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

# =========================================================
# Train / eval
# =========================================================
def train_epoch_ct2pressure(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    l1_loss_fn: nn.Module,
    mse_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    l1_loss_weight: float,
    mse_loss_weight: float,
    grad_loss_weight: float,
    device: str = "cuda",
    epoch_idx: int = 0,
    log_every: int = 20,
) -> Dict[str, float]:
    model.train()

    sums = {
        "total": 0.0,
        "l1": 0.0,
        "mse": 0.0,
        "grad": 0.0,
    }
    n_batches = 0

    total = len(dataloader) if hasattr(dataloader, "__len__") else None
    pbar = tqdm(
        dataloader,
        total=total,
        desc=f"CT2Pressure Train epoch{epoch_idx + 1} batches",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
    )

    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        ct = batch["ct"].to(device)
        target = batch["target"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        pred = model(ct)

        l1_loss = l1_loss_weight * l1_loss_fn(pred, target, valid_mask)
        mse_loss = mse_loss_weight * mse_loss_fn(pred, target, valid_mask)
        grad_loss = grad_loss_weight * grad_loss_fn(pred, target, valid_mask)
        loss = l1_loss + mse_loss + grad_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        sums["total"] += float(loss.item())
        sums["l1"] += float(l1_loss.item())
        sums["mse"] += float(mse_loss.item())
        sums["grad"] += float(grad_loss.item())
        n_batches += 1

        if (log_every > 0) and (n_batches % log_every == 0):
            denom = float(n_batches)
            pbar.set_postfix(
                {
                    "loss": f"{sums['total'] / denom:.4f}",
                    "l1": f"{sums['l1'] / denom:.4f}",
                    "mse": f"{sums['mse'] / denom:.4f}",
                }
            )

    if n_batches == 0:
        return {k: float("nan") for k in sums.keys()}
    return {k: v / n_batches for k, v in sums.items()}


@torch.no_grad()
def eval_epoch_ct2pressure(
    model: nn.Module,
    dataloader: DataLoader,
    l1_loss_fn: nn.Module,
    mse_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    l1_loss_weight: float,
    mse_loss_weight: float,
    grad_loss_weight: float,
    device: str = "cuda",
) -> Dict[str, float]:
    model.eval()

    sums = {
        "total": 0.0,
        "l1": 0.0,
        "mse": 0.0,
        "grad": 0.0,
    }
    n_batches = 0

    for batch in dataloader:
        ct = batch["ct"].to(device)
        target = batch["target"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        pred = model(ct)

        l1_loss = l1_loss_weight * l1_loss_fn(pred, target, valid_mask)
        mse_loss = mse_loss_weight * mse_loss_fn(pred, target, valid_mask)
        grad_loss = grad_loss_weight * grad_loss_fn(pred, target, valid_mask)
        loss = l1_loss + mse_loss + grad_loss

        sums["total"] += float(loss.item())
        sums["l1"] += float(l1_loss.item())
        sums["mse"] += float(mse_loss.item())
        sums["grad"] += float(grad_loss.item())
        n_batches += 1

    if n_batches == 0:
        return {k: float("nan") for k in sums.keys()}
    return {k: v / n_batches for k, v in sums.items()}


# =========================================================
# Args / main
# =========================================================
@dataclass
class Args:
    # -------- data --------
    pressure_dir: str = r"E:\Ning\Projects\predicting acoustic field\data\batch1\pressure"
    ct_dir: str = r"E:\Ning\Projects\predicting acoustic field\data\batch1\ctdata"
    output: str = r"E:\Ning\Projects\predicting acoustic field\results\saved_model_ct2pressure_e2e_new"

    mat_key: str = "amp"
    ct_key: str = "ct_crop"

    seed: int = 42
    batch_size: int = 4
    workers: int = 4
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    normalize_ct: bool = False
    normalize_target: bool = True
    pad_value: float = 0.0

    # -------- model --------
    base_ch: int = 32

    # -------- optimization --------
    epochs: int = 40
    lr: float = 2e-4
    weight_decay: float = 1e-5

    # -------- losses --------
    l1_loss_weight: float = 1.0
    mse_loss_weight: float = 0.3
    grad_loss_weight: float = 0.02
    smooth_l1_beta: float = 1.0

    # -------- debug --------
    max_train_samples: int = 0
    max_val_samples: int = 0
    max_test_samples: int = 0
    log_every: int = 20
    patience: int = 12
    threshold: float = 0.0
    warm_start: int = 10


def main():
    args = Args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files, test_files = split_mat_files(
        args.pressure_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    max_train = args.max_train_samples if args.max_train_samples > 0 else None
    max_val = args.max_val_samples if args.max_val_samples > 0 else None
    max_test = args.max_test_samples if args.max_test_samples > 0 else None

    train_dataset = VxmCTPressurePairDataset(
        ct_dir=Path(args.ct_dir),
        target_paths=train_files,
        ct_key=args.ct_key,
        target_key=args.mat_key,
        normalize_ct=args.normalize_ct,
        normalize_target=args.normalize_target,
        pad_value=args.pad_value,
        max_samples=max_train,
    )
    val_dataset = VxmCTPressurePairDataset(
        ct_dir=Path(args.ct_dir),
        target_paths=val_files,
        ct_key=args.ct_key,
        target_key=args.mat_key,
        normalize_ct=args.normalize_ct,
        normalize_target=args.normalize_target,
        pad_value=args.pad_value,
        max_samples=max_val,
    )
    test_dataset = VxmCTPressurePairDataset(
        ct_dir=Path(args.ct_dir),
        target_paths=test_files,
        ct_key=args.ct_key,
        target_key=args.mat_key,
        normalize_ct=args.normalize_ct,
        normalize_target=args.normalize_target,
        pad_value=args.pad_value,
        max_samples=max_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = CT2PressureResUNet(in_channels=1, out_channels=1, base_ch=args.base_ch).to(device)
    print(f"Model trainable params: {count_trainable_params(model):,}")

    l1_loss_fn = MaskedSmoothL1Loss(beta=args.smooth_l1_beta)
    mse_loss_fn = MaskedMSELoss()
    grad_loss_fn = MaskedSpatialGradientLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    keys = ["total", "l1", "mse", "grad"]
    train_hist = {k: [] for k in keys}
    val_hist = {k: [] for k in keys}

    best_val = float("inf")
    print(f"[CT->Pressure] Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        tr = train_epoch_ct2pressure(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            l1_loss_fn=l1_loss_fn,
            mse_loss_fn=mse_loss_fn,
            grad_loss_fn=grad_loss_fn,
            l1_loss_weight=args.l1_loss_weight,
            mse_loss_weight=args.mse_loss_weight,
            grad_loss_weight=args.grad_loss_weight,
            device=device,
            epoch_idx=epoch,
            log_every=args.log_every,
        )

        va = eval_epoch_ct2pressure(
            model=model,
            dataloader=val_loader,
            l1_loss_fn=l1_loss_fn,
            mse_loss_fn=mse_loss_fn,
            grad_loss_fn=grad_loss_fn,
            l1_loss_weight=args.l1_loss_weight,
            mse_loss_weight=args.mse_loss_weight,
            grad_loss_weight=args.grad_loss_weight,
            device=device,
        )

        for k in keys:
            train_hist[k].append(tr.get(k, float("nan")))
            val_hist[k].append(va.get(k, float("nan")))

        print(
            f"[CT->Pressure] Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train total {tr['total']:.6f} "
            f"(l1 {tr['l1']:.6f}, mse {tr['mse']:.6f}, grad {tr['grad']:.6f}) | "
            f"val total {va['total']:.6f} "
            f"(l1 {va['l1']:.6f}, mse {va['mse']:.6f}, grad {va['grad']:.6f})"
        )

        if va["total"] < best_val:
            best_val = va["total"]
            best_path = weights_dir / "model_ct2pressure_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_total": best_val,
                    "model_state_dict": model.state_dict(),
                    "args": asdict(args),
                },
                best_path,
            )
            print(f"[CT->Pressure] Best checkpoint updated: {best_path} (val total={best_val:.6f})")

        # neurite early_stopping may not always be available here, so use simple patience logic
        if epoch + 1 >= args.warm_start:
            recent = val_hist["total"]
            best_so_far = min(recent)
            best_idx = int(np.argmin(recent))
            if (epoch - best_idx) >= args.patience:
                print(f"[CT->Pressure] Early stopping at epoch {epoch + 1}")
                break

    plot_loss_curves(
        out_dir / "loss_curve" / "ct2pressure_e2e",
        train_hist,
        val_hist,
        stem="model_ct2pressure_e2e",
        keys=keys,
    )

    best_path = weights_dir / "model_ct2pressure_best.pt"
    test_epoch_for_ref = args.epochs - 1
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        test_epoch_for_ref = int(ckpt.get("epoch", test_epoch_for_ref))
        print(f"Loaded best checkpoint for test: {best_path}")
    else:
        print("[WARN] Best checkpoint not found, using current weights for test.")

    test_losses = eval_epoch_ct2pressure(
        model=model,
        dataloader=test_loader,
        l1_loss_fn=l1_loss_fn,
        mse_loss_fn=mse_loss_fn,
        grad_loss_fn=grad_loss_fn,
        l1_loss_weight=args.l1_loss_weight,
        mse_loss_weight=args.mse_loss_weight,
        grad_loss_weight=args.grad_loss_weight,
        device=device,
    )

    print(
        f"[CT->Pressure TEST] total {test_losses['total']:.6f} | "
        f"l1 {test_losses['l1']:.6f} | mse {test_losses['mse']:.6f} | grad {test_losses['grad']:.6f}"
    )

    with open(out_dir / "model_ct2pressure_test_loss.txt", "w", encoding="utf-8") as f:
        for k in keys:
            f.write(f"test_{k}: {test_losses[k]}\n")

    final_bundle = weights_dir / "model_ct2pressure_bundle.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": asdict(args),
            "best_epoch": test_epoch_for_ref,
        },
        final_bundle,
    )
    print(f"Saved model bundle to: {final_bundle}")

    targets_dir = Path(args.pressure_dir)
    target_names = [
        "batch_1_series_1004_image_19.mat",
        "batch_1_series_1007_image_18.mat",
        "batch_1_series_1014_image_14.mat",
        "batch_1_series_1017_image_18.mat",
        "batch_1_series_1027_image_19.mat",
    ]
    picked_targets = [targets_dir / name for name in target_names if (targets_dir / name).exists()]
    if len(picked_targets) > 0:
        viz_png = out_dir / "model_ct2pressure_fixed_5cts.png"
        infer_5cts_and_plot_ct2pressure(
            model=model,
            target_paths=picked_targets,
            ct_dir=Path(args.ct_dir),
            device=device,
            out_png=viz_png,
            ct_key=args.ct_key,
            target_key=args.mat_key,
            normalize_ct=args.normalize_ct,
            normalize_target=args.normalize_target,
            pad_value=args.pad_value,
        )
        print(f"Saved visualization to: {viz_png}")
    else:
        print("[WARN] No picked_targets found for visualization; skip plotting.")


if __name__ == "__main__":
    main()
