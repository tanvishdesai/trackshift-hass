# ============================================================================
# ECOVISION PHASE 2: ATTENTION-AUGMENTED DECODER (HACKATHON EDITION)
# ============================================================================

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import rasterio
from typing import Tuple, Dict, Optional, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ECOVISION PHASE 2: ATTENTION-AUGMENTED DECODER (HACKATHON EDITION)")
print("=" * 80)

# ============================================================================
# SECTION 1: CONFIGURATION (Corrected & Refined)
# ============================================================================

class Config:
    """Central configuration for the training pipeline"""

    # --- Data Paths ---
    ROOT_PATH = "/kaggle/input/onera-satellite-change-detection-dataset"
    IMAGES_PATH = os.path.join(ROOT_PATH, "images", "Onera Satellite Change Detection dataset - Images")
    LABELS_PATH = os.path.join(ROOT_PATH, "train_labels", "Onera Satellite Change Detection dataset - Train Labels")
    OUTPUT_DIR = "/kaggle/working/"
    MODEL_PATH = os.path.join(OUTPUT_DIR, "ecovision_model_v3_attention.pth")
    METRICS_PATH = os.path.join(OUTPUT_DIR, "training_metrics_v3_attention.txt")

    # --- Patch-based training parameters ---
    PATCH_SIZE_INPUT = 256
    PATCH_STRIDE = 128
    MIN_CHANGE_RATIO = 0.01

    # --- Data parameters ---
    IMAGE_SIZE = 256
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    # TEST_SPLIT is the remaining 0.15
    BANDS_TO_USE = ['B04', 'B03', 'B02']  # RGB

    # --- Model parameters (Using a powerful backbone) ---
    USE_PRETRAINED = True
    BACKBONE = "resnet34" # ResNet34 offers a great balance of performance and speed
    TRANSFORMER_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4

    # --- Training Hyperparameters ---
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    # --- FIX: Added the missing attribute to resolve the error ---
    USE_ADAMW = True  # Explicitly define which optimizer is used
    EARLY_STOP_PATIENCE = 15 # More patience as model is more complex

    # --- Loss weights ---
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0
    BCE_WEIGHT = 0.3
    DICE_WEIGHT = 0.4
    FOCAL_WEIGHT = 0.3

    # --- Device & Seeding ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

print(f"Configuration loaded. Training on: {config.DEVICE}")
print(f"Images path: {config.IMAGES_PATH}")
print(f"Labels path: {config.LABELS_PATH}")

# ============================================================================
# SECTION 2: DATASET & AUGMENTATION (Enhanced)
# ============================================================================

class PatchBasedOSCDDataset(Dataset):
    """Dataset class for patch-based change detection."""
    def __init__(self, images_dir: str, labels_dir: str, bands: List[str],
                 patch_size: int, stride: int,
                 min_change_ratio: float, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.bands = bands
        self.patch_size = patch_size
        self.stride = stride
        self.min_change_ratio = min_change_ratio
        self.patches = self._extract_patches()
        if not self.patches:
            raise FileNotFoundError("No patches were extracted. Check data paths and logic.")

    def _load_and_stack_bands(self, band_dir: Path) -> np.ndarray:
        band_arrays = []
        for band in self.bands:
            with rasterio.open(band_dir / f"{band}.tif") as src:
                band_arrays.append(src.read(1))
        return np.stack(band_arrays, axis=0)

    def _extract_patches(self) -> List[Dict]:
        """Extracts patches, balancing change and no-change samples."""
        patches = []
        city_folders = sorted([d for d in self.images_dir.iterdir() if d.is_dir()])
        print(f"\nExtracting patches from {len(city_folders)} cities...")
        for city_dir in city_folders:
            city_name = city_dir.name
            img1_band_dir = city_dir / "imgs_1_rect"
            img2_band_dir = city_dir / "imgs_2_rect"
            label_path = self.labels_dir / city_name / "cm" / f"{city_name}-cm.tif"

            if not (img1_band_dir.exists() and img2_band_dir.exists() and label_path.exists()):
                continue

            img1 = self._load_and_stack_bands(img1_band_dir)
            img2 = self._load_and_stack_bands(img2_band_dir)
            with rasterio.open(label_path) as src:
                mask = src.read(1); mask[mask == 1] = 0; mask[mask == 2] = 1

            _, h, w = img1.shape
            change_patches, no_change_patches = [], []
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    patch_mask = mask[y:y+self.patch_size, x:x+self.patch_size]
                    patch_data = {
                        "img1": img1[:, y:y+self.patch_size, x:x+self.patch_size].copy(),
                        "img2": img2[:, y:y+self.patch_size, x:x+self.patch_size].copy(),
                        "mask": patch_mask.copy()
                    }
                    if patch_mask.sum() / (self.patch_size ** 2) >= self.min_change_ratio:
                        change_patches.append(patch_data)
                    else:
                        no_change_patches.append(patch_data)

            np.random.shuffle(no_change_patches)
            num_to_keep = len(change_patches)
            patches.extend(change_patches)
            patches.extend(no_change_patches[:num_to_keep])
            print(f"  ✓ {city_name}: Extracted {len(change_patches)} change patches and {min(num_to_keep, len(no_change_patches))} no-change patches.")
        print(f"\nTotal patches extracted (balanced): {len(patches)}")
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]
        image1 = torch.from_numpy(patch["img1"]).float()
        image2 = torch.from_numpy(patch["img2"]).float()
        mask = torch.from_numpy(patch["mask"]).float().unsqueeze(0)
        if self.transform:
            image1, image2, mask = self.transform(image1, image2, mask)
        return {"image1": image1, "image2": image2, "mask": mask.squeeze(0)}


class ChangeDetectionTransform:
    """
    HACKATHON UPGRADE: Enhanced data augmentation and preprocessing.
    Includes color jitter and sharpness to handle sensor/seasonal variations.
    """
    def __init__(self, train: bool = True):
        self.train = train
        # Define shared augmentations
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.train:
            # More powerful augmentations for training
            self.geometric_augs = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ])
            self.photometric_augs = T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
            ])

    def __call__(self, img1, img2, mask):
        # Normalize pixel values
        img1 /= 10000.0
        img2 /= 10000.0

        if self.train:
            # Apply geometric augmentations consistently to all inputs
            state = torch.get_rng_state()
            img1 = self.geometric_augs(img1)
            torch.set_rng_state(state)
            img2 = self.geometric_augs(img2)
            torch.set_rng_state(state)
            mask = self.geometric_augs(mask)

            # Apply photometric augmentations independently to each image
            img1 = self.photometric_augs(img1)
            img2 = self.photometric_augs(img2)

        # Apply normalization
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        return img1, img2, mask


def create_dataloaders(config: Config):
    """Create train, validation, and test dataloaders."""
    full_dataset = PatchBasedOSCDDataset(
        images_dir=config.IMAGES_PATH, labels_dir=config.LABELS_PATH, bands=config.BANDS_TO_USE,
        patch_size=config.PATCH_SIZE_INPUT, stride=config.PATCH_STRIDE, min_change_ratio=config.MIN_CHANGE_RATIO
    )
    total_size = len(full_dataset)
    train_size = int(config.TRAIN_SPLIT * total_size)
    val_size = int(config.VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size
    indices = list(range(total_size))
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(full_dataset, train_indices)
    train_dataset.dataset.transform = ChangeDetectionTransform(train=True)
    val_dataset = Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = ChangeDetectionTransform(train=False)
    test_dataset = Subset(full_dataset, test_indices)
    test_dataset.dataset.transform = ChangeDetectionTransform(train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    print(f"\nDataset split complete:\n  - Training: {len(train_dataset)}\n  - Validation: {len(val_dataset)}\n  - Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


# ============================================================================
# SECTION 3: HACKATHON-READY MODEL ARCHITECTURE (Upgraded)
# ============================================================================

print("\n[2/7] Building Attention-Augmented Hybrid Model...")

class ResNetEncoder(nn.Module):
    """Pretrained ResNet for powerful feature extraction."""
    def __init__(self, backbone='resnet34', pretrained=True):
        super().__init__()
        import torchvision.models as models
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.channels = [64, 64, 128, 256, 512]
        else: # Default to resnet18
            resnet = models.resnet18(pretrained=pretrained)
            self.channels = [64, 64, 128, 256, 512]

        self.conv1 = resnet.conv1; self.bn1 = resnet.bn1; self.relu = resnet.relu; self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1; self.layer2 = resnet.layer2; self.layer3 = resnet.layer3; self.layer4 = resnet.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x))) # 1/2
        x1 = self.layer1(self.maxpool(x0))      # 1/4
        x2 = self.layer2(x1)                    # 1/8
        x3 = self.layer3(x2)                    # 1/16
        x4 = self.layer4(x3)                    # 1/32
        return [x0, x1, x2, x3, x4]


class TransformerFusion(nn.Module):
    """Transformer to fuse features from the two temporal images."""
    def __init__(self, dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.proj = nn.Conv2d(512, dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*4, dropout=0.1, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, f1, f2):
        B, _, H, W = f1.shape
        f1, f2 = self.proj(f1), self.proj(f2)
        f1_flat, f2_flat = f1.flatten(2).transpose(1, 2), f2.flatten(2).transpose(1, 2)
        f_concat = torch.cat([f1_flat, f2_flat], dim=1)
        f_fused = self.transformer(f_concat)
        f_diff = torch.abs(f_fused[:, :H*W] - f_fused[:, H*W:])
        return f_diff.transpose(1, 2).reshape(B, -1, H, W)


class AttentionGate(nn.Module):
    """
    HACKATHON UPGRADE: Attention Gate to focus decoder on relevant features.
    This helps suppress noise and highlight salient changes.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    """UNet-style decoder block with optional Attention Gate."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.attention = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=(skip_channels // 2))

    def forward(self, x, skip_connection):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        skip_connection = self.attention(g=x, x=skip_connection)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.relu(self.conv1(x))
        x = self.bn(self.relu(self.conv2(x)))
        return x


class AttentionChangeDetector(nn.Module):
    """
    HACKATHON UPGRADE: Full model with a UNet-style decoder featuring skip connections
    and attention gates for superior segmentation performance.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = ResNetEncoder(config.BACKBONE, config.USE_PRETRAINED)
        self.fusion = TransformerFusion(dim=config.TRANSFORMER_DIM, num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS)
        
        enc_channels = self.encoder.channels
        
        # Decoder blocks
        self.dec4 = DecoderBlock(config.TRANSFORMER_DIM, enc_channels[3], 256)
        self.dec3 = DecoderBlock(256, enc_channels[2], 128)
        self.dec2 = DecoderBlock(128, enc_channels[1], 64)
        self.dec1 = DecoderBlock(64, enc_channels[0], 32)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
        )

    def forward(self, img1, img2):
        # Siamese feature extraction
        feat1_list = self.encoder(img1)
        feat2_list = self.encoder(img2)
        
        # Fuse deepest features with transformer
        fused = self.fusion(feat1_list[-1], feat2_list[-1])
        
        # Feature difference for skip connections
        d3 = torch.abs(feat1_list[3] - feat2_list[3])
        d2 = torch.abs(feat1_list[2] - feat2_list[2])
        d1 = torch.abs(feat1_list[1] - feat2_list[1])
        d0 = torch.abs(feat1_list[0] - feat2_list[0])
        
        # Decode with attention-gated skip connections
        x = self.dec4(fused, d3)
        x = self.dec3(x, d2)
        x = self.dec2(x, d1)
        x = self.dec1(x, d0)
        
        return self.final_conv(x)


# ============================================================================
# SECTION 4: LOSS & METRICS (Unchanged)
# ============================================================================
print("\n[3/7] Setting up loss functions...")
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__(); self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.sigmoid(pred); pred_flat, target_flat = pred.view(-1), target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

class CombinedLoss(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(config.FOCAL_LOSS_ALPHA, config.FOCAL_LOSS_GAMMA)
        self.w_bce = config.BCE_WEIGHT; self.w_dice = config.DICE_WEIGHT; self.w_focal = config.FOCAL_WEIGHT
    def forward(self, pred, target):
        return self.w_bce * self.bce(pred, target) + self.w_dice * self.dice(pred, target) + self.w_focal * self.focal(pred, target)

def calculate_iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(); union = pred.sum() + target.sum() - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).item()

def calculate_metrics(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    tp = (pred_bin * target).sum().item(); fp = (pred_bin * (1 - target)).sum().item()
    tn = ((1 - pred_bin) * (1 - target)).sum().item(); fn = ((1 - pred_bin) * target).sum().item()
    precision = tp / (tp + fp + 1e-6); recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6); accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-6)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}

# ============================================================================
# SECTION 5: TRAINING LOOP (Upgraded Scheduler)
# ============================================================================

print("\n[4/7] Preparing training loop...")

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config: Config):
        self.model = model.to(config.DEVICE)
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.config = config
        self.criterion = CombinedLoss(config)
        
        if config.USE_ADAMW:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        else: # Fallback
            self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # HACKATHON UPGRADE: CosineAnnealingLR is a more robust scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
        
        self.best_iou = 0.0; self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': []}

    def train_epoch(self, epoch):
        self.model.train(); total_loss = 0.0
        for i, batch in enumerate(self.train_loader):
            img1 = batch['image1'].to(self.config.DEVICE); img2 = batch['image2'].to(self.config.DEVICE); mask = batch['mask'].to(self.config.DEVICE).unsqueeze(1)
            self.optimizer.zero_grad()
            output = self.model(img1, img2)
            loss = self.criterion(output, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            print(f"\rEpoch {epoch+1}/{self.config.NUM_EPOCHS} | Batch {i+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}", end='')
        avg_loss = total_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        return avg_loss

    def evaluate(self, loader):
        self.model.eval(); total_loss, total_iou = 0.0, 0.0; all_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
        with torch.no_grad():
            for batch in loader:
                img1 = batch['image1'].to(self.config.DEVICE); img2 = batch['image2'].to(self.config.DEVICE); mask = batch['mask'].to(self.config.DEVICE).unsqueeze(1)
                output = self.model(img1, img2)
                total_loss += self.criterion(output, mask).item()
                total_iou += calculate_iou(output, mask)
                metrics = calculate_metrics(output, mask)
                for k in all_metrics: all_metrics[k] += metrics[k]
        n_batches = len(loader)
        avg_loss = total_loss / n_batches; avg_iou = total_iou / n_batches
        for k in all_metrics: all_metrics[k] /= n_batches
        return avg_loss, avg_iou, all_metrics

    def train(self):
        print("\n" + "="*80 + "\nSTARTING TRAINING\n" + "="*80)
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss, val_iou, metrics = self.evaluate(self.val_loader)
            self.scheduler.step()
            
            self.history['val_loss'].append(val_loss); self.history['val_iou'].append(val_iou); self.history['val_f1'].append(metrics['f1'])
            
            print(f"\n{'='*60}\nEpoch {epoch+1}/{self.config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val IoU: {val_iou:.4f} | Val F1: {metrics['f1']:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if val_iou > self.best_iou:
                self.best_iou = val_iou; self.patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
                print(f"  ✓ New best model saved! IoU: {val_iou:.4f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement. Patience: {self.patience_counter}/{self.config.EARLY_STOP_PATIENCE}")
            
            if self.patience_counter >= self.config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs."); break
        print("\n" + "="*80 + "\nTRAINING COMPLETED\n" + "="*80)
        return self.best_iou

# ============================================================================
# SECTION 6: VISUALIZATION & REPORTING
# ============================================================================
print("\n[5/7] Preparing visualization tools...")
def plot_training_history(history, best_iou, config):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss'); axes[0].plot(epochs, history['val_loss'], 'o-', label='Val Loss')
    axes[0].set_title('Training & Validation Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend()
    axes[1].plot(epochs, history['val_iou'], 'o-', label='Val IoU', color='green')
    axes[1].axhline(y=best_iou, color='r', linestyle='--', label=f'Best IoU: {best_iou:.4f}')
    axes[1].set_title('Validation IoU'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('IoU'); axes[1].legend()
    plt.tight_layout(); save_path = os.path.join(config.OUTPUT_DIR, 'training_history.png')
    plt.savefig(save_path, dpi=150); plt.close(); print(f"✓ Training curves saved to {save_path}")

def visualize_predictions(model, loader, config, num_samples=5):
    model.eval(); samples_shown = 0
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples)); axes = axes.reshape(num_samples, -1)
    with torch.no_grad():
        for batch in loader:
            if samples_shown >= num_samples: break
            img1, img2, mask = batch['image1'].to(config.DEVICE), batch['image2'].to(config.DEVICE), batch['mask'].to(config.DEVICE)
            pred = torch.sigmoid(model(img1, img2)) > 0.5
            for i in range(min(img1.size(0), num_samples - samples_shown)):
                def denorm(img_tensor):
                    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                    return np.clip(img_np * std + mean, 0, 1)
                
                titles = ['Image 1 (Before)', 'Image 2 (After)', 'Ground Truth', 'Prediction']
                images = [denorm(img1[i]), denorm(img2[i]), mask[i].cpu().numpy(), pred[i, 0].cpu().numpy()]
                for j, (title, img) in enumerate(zip(titles, images)):
                    axes[samples_shown, j].imshow(img, cmap='gray' if j >= 2 else None)
                    axes[samples_shown, j].set_title(title); axes[samples_shown, j].axis('off')
                samples_shown += 1
    plt.tight_layout(); save_path = os.path.join(config.OUTPUT_DIR, 'sample_predictions.png')
    plt.savefig(save_path, dpi=150); plt.close(); print(f"✓ Sample predictions saved to {save_path}")

def save_metrics(config, best_iou, final_metrics, history):
    with open(config.METRICS_PATH, 'w') as f:
        f.write("="*80 + "\nECOVISION PHASE 2: ATTENTION-AUGMENTED MODEL METRICS\n" + "="*80 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("KEY HACKATHON UPGRADES:\n  ✓ Attention-Gated UNet-style Decoder\n  ✓ Pretrained ResNet34 Backbone\n")
        f.write("  ✓ Transformer-based Feature Fusion\n  ✓ Enhanced Photometric Augmentations\n  ✓ CosineAnnealingLR Scheduler\n\n")
        f.write("CONFIGURATION:\n  Backbone: {}\n  Batch Size: {}\n  Initial LR: {}\n  Optimizer: {}\n  Epochs Trained: {}\n\n".format(
            config.BACKBONE, config.BATCH_SIZE, config.LEARNING_RATE, 'AdamW' if config.USE_ADAMW else 'Adam', len(history['train_loss'])))
        f.write("FINAL PERFORMANCE (on Test Set):\n  Best Validation IoU: {:.4f}\n  Test IoU: {:.4f}\n  Precision: {:.4f}\n  Recall: {:.4f}\n  F1-Score: {:.4f}\n\n".format(
            best_iou, final_metrics['iou'], final_metrics['precision'], final_metrics['recall'], final_metrics['f1']))
        f.write("TRAINING HISTORY:\n  Final Train Loss: {:.4f}\n  Final Val Loss: {:.4f}\n  Best Val IoU: {:.4f}\n".format(
            history['train_loss'][-1], history['val_loss'][-1], max(history['val_iou'])))
    print(f"✓ Metrics saved to {config.METRICS_PATH}")

# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main():
    try:
        print("\n[STEP 1/5] Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(config)

        print("\n[STEP 2/5] Initializing model...")
        model = AttentionChangeDetector(config)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

        print("\n[STEP 3/5] Setting up trainer...")
        trainer = Trainer(model, train_loader, val_loader, test_loader, config)

        print("\n[STEP 4/5] Training model...")
        best_iou = trainer.train()

        print("\n[STEP 5/5] Final evaluation and visualization...")
        model.load_state_dict(torch.load(config.MODEL_PATH)) # Load best model
        
        print("\n" + "="*80 + "\nPERFORMING FINAL EVALUATION ON UNSEEN TEST SET\n" + "="*80)
        test_loss, test_iou, test_metrics = trainer.evaluate(test_loader)
        test_metrics['iou'] = test_iou
        print(f"Test Set Performance:\n  - Loss: {test_loss:.4f}\n  - IoU: {test_iou:.4f}\n  - F1: {test_metrics['f1']:.4f}")

        save_metrics(config, best_iou, test_metrics, trainer.history)
        plot_training_history(trainer.history, best_iou, config)
        visualize_predictions(model, test_loader, config, num_samples=5)

        print("\n" + "="*80 + "\nPHASE 2 COMPLETED SUCCESSFULLY!\n" + "="*80)
        print(f"📁 Outputs saved in: {config.OUTPUT_DIR}")
        print(f"🎯 Best Validation IoU: {best_iou:.4f}")

    except Exception as e:
        import traceback
        print(f"\n❌ An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()