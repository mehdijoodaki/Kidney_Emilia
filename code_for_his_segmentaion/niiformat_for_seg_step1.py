import os
import glob
import re
import tifffile
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

# ============ CONFIG ============
he_folder = "/beegfs/data/CostaLab/spatial_kidney_kuppelab/Adenin/HE/xenium_separated"
output_base_folder = "/beegfs/data/CostaLab/spatial_kidney_kuppelab/Adenin/HE/output_nifti/"
max_dim = 32767

# hardcoded sample -> (well, position) mapping, from the metadata table
sample_map = {
    "1991-15": ("18", "E4"),
    "1992-15": ("18", "G1"),
    "1993-15": ("18", "G2"),
    "1995-15": ("18", "F2"),
    "1996-15": ("18", "F4"),
    "1997-15": ("18", "F3"),
    "2000-15": ("18", "E1"),
    "2001-15": ("18", "F1"),
    "2003-15": ("18", "C3"),
    "2004-15": ("18", "E3"),
    "2005-15": ("18", "E2"),
    "2006-15": ("18", "D4"),
    "2009-15": ("18", "D3"),
    "2010-15": ("18", "D2"),
    "2013-15": ("18", "D1"),
    "2016-15": ("18", "C1"),
    "2017-15": ("18", "C2"),
    "2018-15": ("18", "C4"),
    "2020-15": ("18", "B4"),
    "2021-15": ("18", "A2 + G3"),
    "2022-15": ("18", "B3 + G4"),
    "2024-15": ("18", "A1"),
    "2025-15": ("18", "B2"),
    "2027-15": ("18", "B1 + H1"),
    "2030-15": ("18", "A4"),
    "2031-15": ("18", "A3"),
}

# only process the samples that actually exist in mice.obs['sample_ID']
#samples_to_process = [
 #   "2031-15", "2022-15", "1993-15", "2018-15",
    #"2021-15", "2024-15", "2016-15", "1991-15",
#]
# =================================
samples_to_process = list(sample_map.keys())

os.makedirs(output_base_folder, exist_ok=True)

not_found = []

for sample_id in samples_to_process:
    if sample_id not in sample_map:
        print(f"WARNING: {sample_id} not in sample_map, skipping")
        continue

    well, positions_raw = sample_map[sample_id]
    positions = [p.strip() for p in positions_raw.split("+")]

    for pos in positions:
        pattern = os.path.join(he_folder, f"{well}-{pos}.ome.tif*")
        matches = [
            f for f in glob.glob(pattern)
            if not os.path.basename(f).startswith(".")
        ]

        if not matches:
            print(f"NOT FOUND: sample={sample_id}, pattern={well}-{pos}.ome.tif*")
            not_found.append((sample_id, well, pos))
            continue

        filepath = matches[0]
        print(f"[{sample_id}] Processing: {filepath}")

        # ---- Read image ----
        img = tifffile.imread(filepath)
        print(f"Original shape: {img.shape}")

        # ---- Select channel if RGB ----
        if img.ndim == 3:
            img_2d = img[..., 0]
        else:
            img_2d = img
        print(f"2D shape: {img_2d.shape}")

        # ---- Resize if too large ----
        if img_2d.shape[0] > max_dim or img_2d.shape[1] > max_dim:
            print("Resizing (too large for NIfTI)...")
            scale_factor = max_dim / max(img_2d.shape)
            new_height = int(img_2d.shape[0] * scale_factor)
            new_width = int(img_2d.shape[1] * scale_factor)
            img_2d = resize(
                img_2d,
                (new_height, new_width),
                preserve_range=True,
                anti_aliasing=True
            )
            print(f"Resized to: {img_2d.shape}")

        # ---- Convert to SimpleITK ----
        sitk_img = sitk.GetImageFromArray(img_2d.astype(np.float32))

        # ---- Build filename ----
        safe_sample_id = re.sub(r"[^\w\-]", "_", sample_id)
        output_name = f"{safe_sample_id}_{well}-{pos}_0000.nii.gz"
        output_path = os.path.join(output_base_folder, output_name)
        sitk.WriteImage(sitk_img, output_path)
        print(f"Saved: {output_path}\n")

print("All files processed ✅")
if not_found:
    print(f"\n{len(not_found)} entries had no matching HE file:")
    for entry in not_found:
        print(entry)
