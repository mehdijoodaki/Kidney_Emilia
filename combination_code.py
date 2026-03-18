
from pilot_gm_vae import *
import scanpy as sc
from pilotpy.plot import *
import requests
import json
import matplotlib.pyplot as plt
import harmonypy as hm


file_path='/data/scRNA/To_Mehdi/Emilia/new_data/combined_human_mice_v0.h5ad'
adata=sc.read_h5ad(file_path)

sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts per cell
sc.pp.log1p(adata)  # Log transform
sc.pp.pca(adata, n_comps=50)  # Compute first 50 principal components

#adata = adata[adata.obs['diagnosis_CK'].isin(['control', 'ICM', 'ICM_AMI', 'DCM'])]

data_mat = adata.obsm['X_pca']  # PCA embeddings
meta_data = adata.obs[['batch']]  # Metadata, e.g., 'batch'
meta_data = pd.DataFrame(meta_data)

vars_use = ['batch']  # Specify the batch column for correction

ho = hm.run_harmony(data_mat, meta_data, vars_use)

adata.obsm['X_pca_harmony'] = ho.Z_corr.T

sc.pp.neighbors(adata,use_rep='X_pca_harmony')
sc.tl.umap(adata)

adata.write_h5ad('/data/scRNA/To_Mehdi/Emilia/new_data/combined_human_mice_v0.h5ad')

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

adata.obsm['X_umap_har'] = adata.obsm['X_umap'].copy()
sc.pp.neighbors(adata, use_rep='z_laten')
sc.tl.umap(adata)
adata.obsm['X_umap_z_laten'] = adata.obsm['X_umap'].copy()

#adata.write_h5ad('/data/scRNA/To_Mehdi/pan_data/xei/filtered_shared_genes_snRNA_XENIUM_object/snRNA_xenium_4922_genes_with_harmony_vae.h5ad')
np.save('/data/scRNA/To_Mehdi/Emilia/new_data/trained_models/snRNA_human_mice/EMD_20_combined.npy', adata.uns['EMD'])
np.save("/data/scRNA/To_Mehdi/Emilia/new_data/trained_models/snRNA_human_mice/z_laten_20_combined.npy", adata.obsm['z_laten'])
np.save("/data/scRNA/To_Mehdi/Emilia/new_data/trained_models/snRNA_human_mice/component_assignment_20_combined.npy", adata.obs['component_assignment'].values)
#adata.write_h5ad('/data/scRNA/To_Mehdi/Emilia/new_data/combined_human_mice_v0.h5ad')


