
from pilot_gm_vae import *
import scanpy as sc
from pilotpy.plot import *
import requests
import json
import matplotlib.pyplot as plt
import harmonypy as hm
import numpy as np


file_path='/data/scRNA/To_Mehdi/pan_data/rebuttal_science/2026-04-10/combined_Rat_Human_Mouse_Embryo.h5ad'
adata=sc.read_h5ad(file_path)
#adata = adata[
 #   ~(
  #      (adata.obs["cell_subtype"] == "Cell cycle LV-CMs") &
   #     (adata.obs["Dataset"] == "embryo")
    #)
#].copy()

adata.X=adata.layers['counts'].copy()

sc.pp.normalize_total(adata, target_sum = 1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, batch_key = 'Dataset', subset = True)
sc.pp.scale(adata, max_value = 10)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
adata.obsm['X_umap_X_pca'] = adata.obsm['X_umap'].copy()
np.save('/data/scRNA/To_Mehdi/pan_data/rebuttal_science/2026-04-10/Cell_hint_all_data/PILOT2/X_umap_X_pca.npy', adata.obsm['X_umap_X_pca'])
np.save('/data/scRNA/To_Mehdi/pan_data/rebuttal_science/2026-04-10/Cell_hint_all_data/PILOT2/X_pca.npy', adata.obsm['X_pca'])

data_mat = adata.obsm['X_pca']  # PCA embeddings
meta_data = adata.obs[['Dataset']]  # Metadata, e.g., 'batch'
meta_data = pd.DataFrame(meta_data)

vars_use = ['Dataset']  # Specify the batch column for correction


ho = hm.run_harmony(data_mat, meta_data, vars_use)

adata.obsm['X_pca_harmony'] = ho.Z_corr.T

sc.pp.neighbors(adata,use_rep='X_pca_harmony')
sc.tl.umap(adata)
adata.obsm['X_umap_X_pca_harmony'] = adata.obsm['X_umap'].copy()
np.save('/data/scRNA/To_Mehdi/pan_data/rebuttal_science/2026-04-10/Cell_hint_all_data/PILOT2/X_umap_X_pca_harmony.npy', adata.obsm['X_umap_X_pca_harmony'])
np.save('/data/scRNA/To_Mehdi/pan_data/rebuttal_science/2026-04-10/Cell_hint_all_data/PILOT2/X_pca_harmony.npy', adata.obsm['X_pca_harmony'])
#adata.write_h5ad('/data/scRNA/To_Mehdi/pan_data/rebuttal_science/2026-04-10/combined_human_pilot2.h5ad')

model = train_gmvae(
    adata=adata,
    dataset_name="snRNA_human_mice",
    pca_key='X_pca_harmony',
    load_weights=False,
    num_classes=15,
    epochs=50
)

gmmvae_wasserstein_distance(
    adata,
    emb_matrix='X_pca_harmony',
    sample_col='Donor',
    status='status',
    num_components=15,
    wass_dis=True,
    apply_gmm=True
)



sc.pp.neighbors(adata, use_rep='z_laten')
sc.tl.umap(adata)
adata.obsm['X_umap_z_laten'] = adata.obsm['X_umap'].copy()

np.save('/data/scRNA/To_Mehdi/pan_data/rebuttal_science/2026-04-10/Cell_hint_all_data/PILOT2/X_umap_z_laten.npy', adata.obsm['X_umap_z_laten'])
np.save('/data/scRNA/To_Mehdi/pan_data/rebuttal_science/2026-04-10/Cell_hint_all_data/PILOT2/z_laten.npy', adata.obsm['z_laten'])

#adata.write_h5ad('/data/scRNA/To_Mehdi/pan_data/rebuttal_science/2026-04-10/combined_Rat_Human_Mouse_Embryo.h5ad')

#np.save('/data/mu0611151/data/mask/Paul/Github/Spatial_Heart/Integration_all_modalities/PILOT_2/PILOT_GM/trained_models/EMD_20_combined.npy', adata.uns['EMD'])
#np.save("/data/mu0611151/data/mask/Paul/Github/Spatial_Heart/Integration_all_modalities/PILOT_2/PILOT_GM/trained_models/z_laten_20_combined.npy", adata.obsm['z_laten'])
#np.save("/data/mu0611151/data/mask/Paul/Github/Spatial_Heart/Integration_all_modalities/PILOT_2/PILOT_GM/trained_models/component_assignment_20_combined.npy", adata.obs['component_assignment'].values)



