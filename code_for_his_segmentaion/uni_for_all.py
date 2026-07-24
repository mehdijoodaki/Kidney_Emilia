import torch
import os
from PIL import Image
import pandas as pd
import numpy as np
import re
import glob
import tifffile
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ CONFIG ============
he_folder = "/beegfs/data/CostaLab/spatial_kidney_kuppelab/Adenin/HE/xenium_separated"
output_base_folder = "/beegfs/data/CostaLab/spatial_kidney_kuppelab/Adenin/HE/output_patches/"
metadata_file = "/beegfs/data/CostaLab/spatial_kidney_kuppelab/Adenin/HE/output_patches_meta.csv"
embeddings_file = "/beegfs/data/CostaLab/spatial_kidney_kuppelab/Adenin/HE/UNI_patch_embeddings_224.csv"

local_dir = "/beegfs/data/CostaLab/SpatialHeart/Histology_Xenium_human/uni_weight/"

patch_size = (224, 224)
stride = 224
# =================================

os.makedirs(output_base_folder, exist_ok=True)

# ---------- Load UNI model ----------
timm_kwargs = {
    'model_name': 'vit_giant_patch14_224',
    'img_size': 224,
    'patch_size': 14,
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5,
    'embed_dim': 1536,
    'mlp_ratio': 2.66667 * 2,
    'num_classes': 0,
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked,
    'act_layer': torch.nn.SiLU,
    'reg_tokens': 8,
    'dynamic_img_size': True
}
model = timm.create_model(**timm_kwargs)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
model.eval()
model.to(device)

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# ---------- Step 1: Patch extraction for every HE file in the folder ----------
failed_files = []
all_patch_metadata = []

he_files = sorted([
    f for f in glob.glob(os.path.join(he_folder, "*.ome.tif*"))
    if os.path.basename(f).startswith("18") or True  # keep simple: just take every .ome.tif* file
])

# if you only want files literally starting with "HE", use this instead:
# he_files = sorted([
#     f for f in glob.glob(os.path.join(he_folder, "HE*.ome.tif*"))
#     if not os.path.basename(f).startswith(".")
# ])

print(f"Found {len(he_files)} HE files in {he_folder}")

for filepath in he_files:
    try:
        wsi_image = tifffile.imread(filepath)
        wsi_image = Image.fromarray(wsi_image)

        slide_name = os.path.basename(filepath).replace(".ome.tif", "").replace(".tiff", "").replace(".tif", "")
        slide_name = re.sub(r"[^\w\-]", "_", slide_name)

        slide_output_folder = os.path.join(output_base_folder, slide_name)
        os.makedirs(slide_output_folder, exist_ok=True)

        wsi_width, wsi_height = wsi_image.size
        print(f"Processing {slide_name} - Size: {wsi_width}x{wsi_height}")

        saved_patches = 0
        for x in range(0, wsi_width - patch_size[0], stride):
            for y in range(0, wsi_height - patch_size[1], stride):
                patch = wsi_image.crop((x, y, x + patch_size[0], y + patch_size[1]))

                patch_filename = f"patch_{saved_patches+1}.png"
                patch_path = os.path.join(slide_output_folder, patch_filename)
                patch.save(patch_path)

                all_patch_metadata.append([slide_name, patch_filename, x, y, patch_path])
                saved_patches += 1

        print(f"Saved {saved_patches} patches for {slide_name}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        failed_files.append(filepath)

metadata_df = pd.DataFrame(
    all_patch_metadata,
    columns=['Slide_ID', 'Patch_ID', 'X', 'Y', 'Path']
)
metadata_df.to_csv(metadata_file, index=False)

print("Patch extraction done!")
print("Failed files:", failed_files)
print(f"Metadata saved in {metadata_file}")

metadata_df = pd.read_csv(metadata_file).reset_index(drop=True)


# ---------- Step 2: Feature extraction ----------
class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, "*.png"))
        self.transform = transform
        self.patch_ids = [os.path.basename(img) for img in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, -1


root_path = output_base_folder
valid_folders = [
    f for f in sorted(os.listdir(root_path))
    if os.path.isdir(os.path.join(root_path, f)) and not f.startswith(".")
]

all_embeddings = []
all_sample_ids = []
all_patch_ids = []

for folder in valid_folders:
    folder_path = os.path.join(root_path, folder)

    test_dataset = PatchDataset(root_dir=folder_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    test_features = extract_patch_features_from_dataloader(model, test_dataloader)
    test_feats = torch.Tensor(test_features['embeddings'])

    all_embeddings.append(test_feats.numpy())
    all_sample_ids.extend([folder] * test_feats.shape[0])
    all_patch_ids.extend(test_dataset.patch_ids)

df_embeddings = pd.DataFrame(
    torch.cat([torch.tensor(arr) for arr in all_embeddings], dim=0).numpy()
)
df_embeddings["Slide_ID"] = all_sample_ids
df_embeddings["Patch_ID"] = all_patch_ids

df_embeddings['match_id'] = df_embeddings['Slide_ID'] + '_' + df_embeddings['Patch_ID']
metadata_df['match_id'] = metadata_df['Slide_ID'] + '_' + metadata_df['Patch_ID']
df_embeddings = df_embeddings.merge(metadata_df[['match_id', 'X', 'Y', 'Patch_ID']], on='match_id')
df_embeddings.to_csv(embeddings_file, index=False)

print(f"Final embeddings saved to {embeddings_file}")
