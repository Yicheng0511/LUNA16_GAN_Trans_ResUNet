import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
from tqdm import tqdm

ANNOTATION_PATH = "Lungs_Segment_Info_Folder/annotations.csv"
IMG_DIR = "Lungs_CT_Dataset"
SAVE_DIR = "Lungs_POS_NPY_25d"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_mhd(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    itk = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(itk)
    origin = np.array(itk.GetOrigin())
    spacing = np.array(itk.GetSpacing())
    direction = np.array(itk.GetDirection())
    del itk
    return array, origin, spacing, direction

def normalize(img: np.ndarray) -> np.ndarray:
    min_hu = -1000
    max_hu = 400
    img = np.clip(img, min_hu, max_hu)
    img = (img-min_hu) / (max_hu-min_hu)
    return img.astype(np.float32)

def world_to_pixel(x_pos: float, y_pos: float, z_pos: float, origin: np.ndarray, spacing: np.ndarray, direction: np.ndarray) -> tuple[float, float, float]:
    dx, dy, dz = direction[0], direction[4], direction[8]
    pixel_x_pos = int(round((x_pos-origin[0]) / spacing[0] * dx))
    pixel_y_pos = int(round((y_pos-origin[1]) / spacing[1] * dy))
    pixel_z_pos = int(round((z_pos-origin[2]) / spacing[2] * dz))
    return pixel_x_pos, pixel_y_pos, pixel_z_pos

def create_mask_2d(shape: tuple[int, int], x_pos: float, y_pos: float, radius: float, x_spacing: float, y_spacing:float) -> np.ndarray:
    mask = np.zeros(shape, np.float32)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    distance = np.sqrt(((xx-x_pos)*x_spacing)**2 + ((yy-y_pos)*y_spacing)**2)
    mask[distance <= radius] = 1
    return mask

data_file = pd.read_csv(ANNOTATION_PATH)
positive_list = []

for idx, row in tqdm(data_file.iterrows(), total=len(data_file), desc="Preprocessing Positive Samples"):
    seriesuid = row["seriesuid"]
    fname = f"{seriesuid}.mhd"
    path = os.path.join(IMG_DIR, fname)
    
    if not os.path.exists(path):
        continue

    ct, origin, spacing, direction = load_mhd(path)
    x, y, z = world_to_pixel(
        x_pos=row.coordX,
        y_pos=row.coordY,
        z_pos=row.coordZ,
        origin=origin,
        spacing=spacing,
        direction=direction
    )

    if 0 <= z < ct.shape[0]:
        z_center = int(round(z))
        depth = ct.shape[0]

        z_slices = []
        for dz in [-1, 0, 1]:
            curr_z = z_center + dz
            curr_z = max(0, min(curr_z, depth - 1))
            z_slices.append(curr_z)
            
        img_25d = np.stack([normalize(ct[zi]) for zi in z_slices], axis=0)
        
        radius = max(row.diameter_mm / 2, 2)
        mask_slice = create_mask_2d(
            shape=ct[z_center].shape,
            x_pos=x,
            y_pos=y,
            radius=radius,
            x_spacing=spacing[0],
            y_spacing=spacing[1]
        )

        base_path = os.path.join(SAVE_DIR, f"pos_{seriesuid}_{idx}")
        np.save(f"{base_path}_img.npy", img_25d)
        np.save(f"{base_path}_mask.npy", mask_slice)

        positive_list.append({
            "file_path": base_path,
            "is_negative": False
        })

np.save(os.path.join(SAVE_DIR, "positive_list.npy"), positive_list)
print(f"Proprocessed frinished! Total: {len(positive_list)} slices")
