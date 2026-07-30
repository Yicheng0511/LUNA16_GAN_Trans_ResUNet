import SimpleITK as sitk
import torch
import random
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


class LungsDataset(torch.utils.data.Dataset):
    def __init__(self, file_list: tuple[dict, ...], dataset_dir: Path, transform=None) -> None:
        self.file_list = file_list
        self.transform = transform
        self.cache = {}

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        info = self.file_list[idx]
        if info["is_negative"]:
            file_name = info["file_path"]
            z_center = info["z"]
            cache_key = f"{file_name}_{z_center}"
            if cache_key not in self.cache:
                ct = self.load_mhd_with_info(file_name)
                img = ct[z_center].astype(np.float32)
                depth = ct.shape[0]
                z_slices = []
                for dz in [-1, 0, 1]:
                    curr_z = z_center + dz
                    curr_z = max(0, min(curr_z, depth - 1))
                    z_slices.append(curr_z)

                img_25d = np.stack([self.normalize_slice(ct[zi]) for zi in z_slices], axis=0)
                self.cache[cache_key] = img_25d
                del ct
            else:
                img_25d = self.cache[cache_key]
                
            mask_shape = (1,) + img_25d.shape[1:]
            mask = np.zeros(mask_shape, np.float32)

        else:
            npy_path = info["file_path"]
            img_25d = np.load(f"{npy_path}_img.npy")
            mask = np.load(f"{npy_path}_mask.npy")
            mask = np.expand_dims(mask, axis=0)

        img_np = np.transpose(img_25d, (1, 2, 0))
        mask_np = mask[0]
        
        if self.transform is not None:
            aug_out = self.transform(image=img_np, mask=mask_np)
            img_tensor = aug_out["image"].float()
            mask_tensor = aug_out["mask"].unsqueeze(0).float()
        else:
            img_tensor = torch.from_numpy(img_25d).float()
            mask_tensor = torch.from_numpy(mask).float()
        
        return img_tensor, mask_tensor

    @staticmethod
    def load_mhd_with_info(path: Path) -> np.ndarray:
        image = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(image)
        del image
        return array
    
    @staticmethod
    def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
        min_hu = -1000
        max_hu = 400
        slice_2d = slice_2d
        slice_2d = np.clip(slice_2d, min_hu, max_hu)
        slice_2d = (slice_2d-min_hu) / (max_hu-min_hu)
        return slice_2d
