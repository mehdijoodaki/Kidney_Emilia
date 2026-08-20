
from pilot_gm_vae import *
import scanpy as sc
from pilotpy.plot import *
import requests
import json
import matplotlib.pyplot as plt
import harmonypy as hm
import numpy as np


file_path='/data/scRNA/To_Mehdi/Emilia/flex/Integration_mouse_human.h5ad'
adata=sc.read_h5ad(file_path)

adata.X=adata.layers['counts'].copy()

sc.pp.normalize_total(adata, target_sum = 1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, batch_key = 'Dataset', subset = True)
sc.pp.scale(adata, max_value = 10)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
adata.obsm['X_umap_X_pca'] = adata.obsm['X_umap'].copy()
np.save('/data/scRNA/To_Mehdi/Emilia/flex/pilot2_embeddings/X_umap_X_pca.npy', adata.obsm['X_umap_X_pca'])
np.save('/data/scRNA/To_Mehdi/Emilia/flex/pilot2_embeddings/X_pca.npy', adata.obsm['X_pca'])

data_mat = adata.obsm['X_pca']  # PCA embeddings
meta_data = adata.obs[['Dataset']]  # Metadata, e.g., 'batch'
meta_data = pd.DataFrame(meta_data)

vars_use = ['Dataset']  # Specify the batch column for correction


ho = hm.run_harmony(data_mat, meta_data, vars_use)

adata.obsm['X_pca_harmony'] = ho.Z_corr.T

sc.pp.neighbors(adata,use_rep='X_pca_harmony')
sc.tl.umap(adata)
adata.obsm['X_umap_X_pca_harmony'] = adata.obsm['X_umap'].copy()
np.save('/data/scRNA/To_Mehdi/Emilia/flex/pilot2_embeddings/X_umap_X_pca_harmony.npy', adata.obsm['X_umap_X_pca_harmony'])
np.save('/data/scRNA/To_Mehdi/Emilia/flex/pilot2_embeddings/X_pca_harmony.npy', adata.obsm['X_pca_harmony'])


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

np.save('/data/scRNA/To_Mehdi/Emilia/flex/pilot2_embeddings/X_umap_z_laten.npy', adata.obsm['X_umap_z_laten'])
np.save('/data/scRNA/To_Mehdi/Emilia/flex/pilot2_embeddings/z_laten.npy', adata.obsm['z_laten'])





