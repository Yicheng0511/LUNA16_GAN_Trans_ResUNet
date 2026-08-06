import numpy as np
import time
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
import gc
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import draw_segmentation_masks
from torch_ema import ExponentialMovingAverage
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from model_framework.Trans_ResUNet3plus import ResUNet3plus as generator
from model_framework.discriminator import ResPatchGANDiscriminator as discriminator
from datasets.dataset_25d import LungsDataset
from loss_function.loss import TverskyFocalLoss
from metrics.metrics import Dice, Precision, Recall

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if torch.cuda.is_available():
    try:
        from torch.amp import autocast
    except ImportError:
        from torch.cuda.amp import autocast
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

annotation_path: Path = Path(config["paths"]["annotation_path"])
dataset_dir: Path = Path(config["paths"]["dataset_dir"])
model_path: Path = Path(config["paths"]["model_path"])
npy_dir: Path = Path(config["paths"]["npy_dir"])
log_dir: Path = Path(config["paths"]["log_dir"])

patience: int = config["early_stopping"]["patience"]
    
loss_alpha: float = config["loss"]["alpha"]
loss_beta: float = config["loss"]["beta"]
loss_gamma: float = config["loss"]["gamma"]
weight_adv: float = config["loss"]["weight_adv"]

output_weights: tuple[float, ...] = tuple(config["output_weights"])

batch: int = config["train"]["batch"]
epoch: int = config["train"]["epoch"]
cnn_lr: float = config["train"]["cnn_lr"]
trans_lr: float = config["train"]["trans_lr"]
d_lr: float = config["train"]["d_lr"]
cnn_weight_decay: float = config["train"]["cnn_weight_decay"]
trans_weight_decay: float = config["train"]["trans_weight_decay"]
negative_ratio: float = config["train"]["negative_ratio"]
g_iter: int = config["train"]["g_iter"]
d_iter: int = config["train"]["d_iter"]
base_ch: int = config["train"]["base_ch"]

writer = SummaryWriter(log_dir=log_dir)
ann = pd.read_csv(annotation_path)
mhd_files = tuple(dataset_dir.glob("*.mhd"))
    
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.4),
    A.RandomRotate90(p=0.5),
    ToTensorV2()
])

    
class EarlyStopping:
    def __init__(self, patience, threshold=0.0001):
        self.patience = patience
        self.threshold = threshold
        self.best_value = -1.0
        self.counter = 0
        self.early_stop = False

    def __call__(self, value):
        if value > self.best_value + self.threshold:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


def calculate_fp_fn(y_pred, y_true, threshold=0.4):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred >= threshold).float()
    y_pred = y_pred.reshape(-1)
    y_true = y_true.float()
    y_true = y_true.reshape(-1)

    fp = torch.sum(y_pred * (1-y_true))
    fn = torch.sum((1-y_pred) * y_true)

    return fp, fn

def overlay_pred_on_ct(ct_img: torch.Tensor, pred_mask: torch.Tensor, colors, alpha=0.5):
    ct_rgb = (ct_img * 255).repeat(3,1,1).to(torch.uint8)
    overlay = draw_segmentation_masks(
        image=ct_rgb,
        masks=pred_mask.bool(),
        alpha=alpha,
        colors=colors
    )
    
    return overlay


def overlay_pred_gt_only(gt_mask, pred_mask, threshold=0.4):
    """
    gt_mask: [1, H, W] binary ground truth tensor
    pred_mask: [H, W] binary prediction mask (after sigmoid & threshold)
    return RGB image tensor [3, H, W] without CT background
    """
    
    gt_bin = (gt_mask > threshold).float()[0]
    pred_bin = pred_mask.float()
    H, W = gt_bin.shape
    
    rgb = torch.zeros((3, H, W), device=gt_mask.device)

    rgb[0] = pred_bin
    rgb[1] = gt_bin

    return rgb


def train_one_epoch(G, D, criterion, opt_G, opt_D, train_loader, epoch, device, ema_G):
    epoch_loss_D = torch.tensor(0.0, device=device)
    epoch_loss_G = torch.tensor(0.0, device=device)
    epoch_dice = torch.tensor(0.0, device=device)
    epoch_precision = torch.tensor(0.0, device=device)
    epoch_recall = torch.tensor(0.0, device=device)
    epoch_fp = torch.tensor(0.0, device=device)
    epoch_fn = torch.tensor(0.0, device=device)
    G.train()
    D.train()
    opt_D.zero_grad(set_to_none=True)
    opt_G.zero_grad(set_to_none=True)
    for train_index, (images, masks) in enumerate(train_loader):
        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(masks)):
            continue
        images = images.to(device)
        masks = masks.to(device)
        
        for _ in range(d_iter):
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    fake_masks = torch.sigmoid(G(images)[0]).detach()
                real_in = torch.cat([images, masks], dim=1)
                fake_in = torch.cat([images, fake_masks], dim=1)
                real_pred = D(real_in)
                fake_pred = D(fake_in)

                real_score = F.binary_cross_entropy_with_logits(real_pred.float(), torch.full_like(real_pred, 0.8))
                fake_score = F.binary_cross_entropy_with_logits(fake_pred.float(), torch.full_like(fake_pred, 0.2))
                loss_D = (real_score+fake_score) / 2
                loss_scaled_D = loss_D / accumulate_steps

            loss_scaled_D.backward()

            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1)
            opt_D.step()
            opt_D.zero_grad(set_to_none=True)
            
        with torch.no_grad():    
            epoch_loss_D += loss_D.detach()
        del loss_D, fake_masks, real_pred, fake_pred, real_score, fake_score, real_in, fake_in, loss_scaled_D

        last_loss_G = None
        
        for _ in range(g_iter):
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = G(images)            
                fake_in_G = torch.cat([images, torch.sigmoid(outputs[0])], dim=1)
                fake_pred_for_G = D(fake_in_G)

                loss_seg = sum(weight * criterion(out.float(), masks.float()) for out, weight in zip(outputs, output_weights))
                loss_adv = F.binary_cross_entropy_with_logits(
                    fake_pred_for_G.float(),
                    torch.full_like(fake_pred_for_G, 0.8, dtype=torch.float32)
                )

                loss_G = loss_seg + weight_adv * loss_adv
                loss_scaled_G = loss_G / accumulate_steps
            
            loss_scaled_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1)
            opt_G.step()
            ema_G.update()
            opt_G.zero_grad(set_to_none=True)
            last_loss_G = loss_G.detach()
                
        with torch.no_grad():
            pred = G(images)[0].float()
            fp, fn = calculate_fp_fn(pred, masks.float())
            epoch_loss_G += last_loss_G.detach()
            epoch_dice += Dice(pred, masks.float())
            epoch_precision += Precision(pred, masks.float())
            epoch_recall += Recall(pred, masks.float())
            epoch_fp += fp
            epoch_fn += fn

        del images, masks, outputs, loss_seg, loss_adv, loss_G, fake_in_G, fake_pred_for_G, loss_scaled_G

    return epoch_loss_D, epoch_loss_G, epoch_dice, epoch_precision, epoch_recall, epoch_fp, epoch_fn
    
    
def validation(model, criterion, val_loader, device):
    val_loss = torch.tensor(0.0, device=device)
    val_dice = torch.tensor(0.0, device=device)
    val_precision = torch.tensor(0.0, device=device)
    val_recall = torch.tensor(0.0, device=device)
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                total_loss = sum(weight * criterion(out, masks) for out, weight in zip(outputs, output_weights))
                
            pred = outputs[0].detach()
            masks_detach = masks.detach()
            val_dice += Dice(pred, masks_detach)
            val_precision += Precision(pred, masks.float())
            val_recall += Recall(pred, masks.float())
            val_loss += total_loss.detach()
            del images, masks, outputs, total_loss, pred, masks_detach
            
    return val_loss, val_dice, val_precision, val_recall
    

def train_model(G, D, criterion, optimizer_G, optimizer_D, scheduler_G, scheduler_D, train_loader, val_loader, device, num_epochs, ema_G):
    total_start_time = time.time()
    early_stopping = EarlyStopping(patience=patience)
    best_val_dice = 0.0
    
    G.eval()
    with torch.no_grad():
        val_images, val_masks = next(iter(val_loader))
        first_val_file_info = val_dataset.file_list[0]
        print("First validation sample path:", first_val_file_info["file_path"])
        val_images = val_images.to(device)
        val_masks = val_masks.to(device)
            
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        epoch_loss_D, epoch_loss_G, epoch_dice, epoch_precision, epoch_recall, epoch_fp, epoch_fn = train_one_epoch(G, D, criterion, optimizer_G, optimizer_D, train_loader, epoch, device, ema_G)
        with ema_G.average_parameters():
            val_loss, val_dice, val_precision, val_recall = validation(G, criterion, val_loader, device)

        final_stack = torch.stack([
            epoch_loss_D,
            epoch_loss_G,
            epoch_dice,
            val_loss,
            val_dice,
            epoch_fp,
            epoch_fn,
            epoch_precision,
            epoch_recall,
            val_precision,
            val_recall
        ]).cpu()
        
        del epoch_loss_D, epoch_loss_G, epoch_dice, val_loss, val_dice, epoch_fp, epoch_fn, epoch_precision, epoch_recall, val_precision, val_recall
        
        epoch_loss_D = final_stack[0].item() / len(train_loader)
        epoch_loss_G = final_stack[1].item() / len(train_loader)
        epoch_dice = final_stack[2].item() / len(train_loader)
        val_loss = final_stack[3].item() / len(val_loader)
        val_dice = final_stack[4].item() / len(val_loader)
        epoch_fp = final_stack[5].item()
        epoch_fn = final_stack[6].item()
        epoch_precision = final_stack[7].item() / len(train_loader)
        epoch_recall = final_stack[8].item() / len(train_loader)
        val_precision = final_stack[9].item() / len(val_loader)
        val_recall = final_stack[10].item() / len(val_loader)
        
        if scheduler_G is not None:
            scheduler_G.step(val_dice)
            
        if scheduler_D is not None:
            scheduler_D.step()
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        gc.collect()
        torch.cuda.empty_cache()
        
        writer.add_scalar("Loss_D/Train", epoch_loss_D, epoch)
        writer.add_scalar("Loss_G/Train", epoch_loss_G, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Metric/Dice_Train", epoch_dice, epoch)
        writer.add_scalar("Metric/Dice_Val", val_dice, epoch)
        writer.add_scalar("LR/G_LR", optimizer_G.param_groups[0]['lr'], epoch)
        writer.add_scalar("LR/D_LR", optimizer_D.param_groups[0]['lr'], epoch)
        writer.add_scalar("False/FP", epoch_fp, epoch)
        writer.add_scalar("False/FN", epoch_fn, epoch)
        writer.add_scalar("Metric/Epoch_Precision", epoch_precision, epoch)
        writer.add_scalar("Metric/Epoch_Recall", epoch_recall, epoch)
        writer.add_scalar("Metric/Val_Precision", val_precision, epoch)
        writer.add_scalar("Metric/Val_Recall", val_recall, epoch)
        
        G.eval()
        with torch.no_grad():
            pred1, *_ = G(val_images[:1])
            pred1 = torch.sigmoid(pred1[:1])
            ct_image = val_images[0, 1].unsqueeze(0)
            pred_mask = (pred1[0, 0].cpu() > 0.45)
            if epoch == 0:
                writer.add_image("CT/Input", ct_image, epoch)
                writer.add_image("CT/GroundTruth", overlay_pred_on_ct(ct_image, val_masks[0], (0, 255, 0)), epoch)

            writer.add_image("CT/Prediction", overlay_pred_on_ct(ct_image, pred_mask, (255, 0, 0)), epoch)
            writer.add_image("CT/Prediction_NO_bg", pred1[0], epoch)
            writer.add_image("CT/Prediction_on_mask", overlay_pred_gt_only(val_masks[0], pred_mask), epoch)
        G.train()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Duration: {epoch_duration:.2f} seconds, '
            f'Loss_D: {epoch_loss_D:.4f}, Loss_G: {epoch_loss_G:.4f}, Validation Loss: {val_loss:.4f}, '
            f'dice Coefficient: {epoch_dice:.4f}, Validation dice: {val_dice:.4f}')
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            with ema_G.average_parameters():
                torch.save(G.state_dict(), model_path)
            print(f"Best model saved (Val dice: {best_val_dice:.4f})")
        
        if early_stopping(val_dice):
            print("Early Stopping Activated! Training Stopped")
            break

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f'Best Validation Dice: {best_val_dice:.4f}')
    print(f'Total training time: {total_duration:.2f} seconds')


def predict(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        sup1, *_ = model(image_tensor)
        pred = torch.sigmoid(sup1)
    return pred


def add_negative_samples(positive_list: list[dict], neg_pool: tuple[Path, ...], negative_ratio: float) -> tuple[dict, ...]:
    pos_len = len(positive_list)
    neg_len = int(pos_len * negative_ratio)
    
    if negative_ratio == 0:
        return tuple(positive_list)
    
    rng = np.random.default_rng(seed=42)
    neg_samples: list = []
    
    sample_num = min(neg_len, len(neg_pool))
    selected_paths = rng.choice(neg_pool, size=sample_num, replace=False)
    
    for file_path in selected_paths:
        ct = LungsDataset.load_mhd_with_info(file_path)
        z = rng.integers(ct.shape[0])
        neg_samples.append({"file_path": file_path, "z": z, "is_negative": True})
    return tuple(positive_list + neg_samples)


def get_param_groups(model):
    cnn_decay = []
    trans_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        skip_wd_keywords = ["bias", "norm", "pos_embed", "cls_token", "relative_position_bias_table"]
        if any(k in name for k in skip_wd_keywords):
            no_decay.append(param)
            continue

        if name.startswith(("e1.", "e2.", "e3.", "e4.")):
            cnn_decay.append(param)
        elif name.startswith((
            "bottleneck.",
            "fusion.",
            "conv1.", "conv2.", "conv3.", "conv4.", "conv5.", "conv_dec.",
            "out1.", "out2.", "out3.", "out4.", "out5."
        )):
            trans_decay.append(param)
        else:
            trans_decay.append(param)
    
    groups = [
        {"params": cnn_decay, "lr": cnn_lr, "base_lr": cnn_lr, "weight_decay": cnn_weight_decay},
        {"params": trans_decay, "lr": trans_lr, "base_lr": trans_lr, "weight_decay": trans_weight_decay},
        {"params": no_decay, "lr": trans_lr, "base_lr": trans_lr, "weight_decay": 0.0},
    ]
    return groups


def get_discriminator_param_groups(d_model, weight_decay):
    decay_params = []
    no_decay_params = []
    for name, param in d_model.named_parameters():
        if not param.requires_grad:
            continue
        if "conv" in name and "weight" in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = generator().to(device=device)
    D = discriminator().to(device=device)
    param_groups_G = get_param_groups(G)
    d_wd = 0.5 * cnn_weight_decay
    param_groups_D = get_discriminator_param_groups(D, d_wd)
    criterion = TverskyFocalLoss(alpha=loss_alpha, beta=loss_beta, gamma=loss_gamma)

    optimizer_G = torch.optim.AdamW(param_groups_G, betas=(0.9, 0.999))
    optimizer_D = torch.optim.AdamW(param_groups_D, lr=d_lr, betas=(0.9, 0.999))
    
    scheduler_G = ReduceLROnPlateau(
        optimizer_G,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=2.5e-6,
        threshold=0.002,
        threshold_mode="abs",
        cooldown=2
    )
    
    scheduler_D = None

    positive_samples = np.load(npy_dir / "positive_list.npy", allow_pickle=True).tolist()
    patient_dict = defaultdict(list)

    for item in positive_samples:
        fname = Path(item["file_path"]).name
        pid = fname.rsplit("_", 1)[0]
        patient_dict[pid].append(item)

    full = list(patient_dict.keys())
    split_rng = np.random.default_rng(seed=42)
    split_rng.shuffle(full)
    train_pids, pos_temp = train_test_split(full, test_size=0.2, random_state=42)
    val_pids, test_pids = train_test_split(pos_temp, test_size=0.5, random_state=42)

    train_files = [item for pid in train_pids for item in patient_dict[pid]]
    val_files = [item for pid in val_pids for item in patient_dict[pid]]
    test_files = [item for pid in test_pids for item in patient_dict[pid]]
    all_files = train_files + val_files + test_files
    print("Original positive sample:", len(train_files))
    
    pos_file_set = {item["file_path"] for item in all_files}
    neg_pool = [path for path in mhd_files if path.name not in pos_file_set]
    
    split_rng.shuffle(neg_pool)
    train_neg_pool, neg_temp = train_test_split(neg_pool, test_size=0.2, random_state=42)
    val_neg_pool, test_neg_pool = train_test_split(neg_temp, test_size=0.5, random_state=42)

    train_files = add_negative_samples(positive_list=train_files, neg_pool=train_neg_pool, negative_ratio=negative_ratio)
    val_files = add_negative_samples(positive_list=val_files, neg_pool=val_neg_pool, negative_ratio=negative_ratio)
    test_files = add_negative_samples(positive_list=test_files, neg_pool=test_neg_pool, negative_ratio=negative_ratio)

    train_dataset = LungsDataset(file_list=train_files, dataset_dir=dataset_dir, transform=train_transform)
    val_dataset = LungsDataset(file_list=val_files, dataset_dir=dataset_dir)
    test_dataset = LungsDataset(file_list=test_files, dataset_dir=dataset_dir)
    
    print("Total Training Dataset:", len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch,
        shuffle=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch,
        shuffle=False,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch,
        shuffle=False,
        drop_last=True
    )

    print("Total Training Dataset:", len(train_dataset))

    if os.path.exists(model_path):
        G.load_state_dict(torch.load(
            model_path,
            map_location=device,
            weights_only=True
        ))
        
        print("Model loaded successfully from checkpoint.")
    ema_G = ExponentialMovingAverage(G.parameters(), decay=0.996)
    train_model(
        G=G,
        D=D,
        criterion=criterion,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        scheduler_G=scheduler_G,
        scheduler_D=scheduler_D,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=epoch,
        ema_G=ema_G
    )
