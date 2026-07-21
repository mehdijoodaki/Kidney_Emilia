import scanpy as sc
import pandas as pd
import anndata as ad

# ---------------- config ----------------
CSV_PATH = "UNI_patch_embeddings_224.csv"
OUT_PATH = "all_his.h5ad"

N_PCS = 50
RESOLUTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
N_NEIGHBORS = 15
# -----------------------------------------

df = pd.read_csv(CSV_PATH)

emb_cols = [c for c in df.columns if c.isdigit()]
obs_cols = [c for c in df.columns if c not in emb_cols]

adata = ad.AnnData(X=df[emb_cols].values, obs=df[obs_cols].copy())

# -- scanpy-native PCA (zero-centers only, no unit-variance scaling) --
#sc.pp.pca(adata, n_comps=N_PCS)

# -- alternative, matching the merge script (StandardScaler + sklearn PCA): --
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X_scaled = StandardScaler().fit_transform(adata.X)
adata.obsm["X_pca"] = PCA(n_components=N_PCS, random_state=0).fit_transform(X_scaled)

sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, use_rep="X_pca")
for res in RESOLUTIONS:
    key = f"leiden_{str(res).replace('.', '')}"
    sc.tl.leiden(adata, resolution=res, key_added=key)
sc.tl.umap(adata)

adata.write_h5ad(OUT_PATH)
print(f"saved {OUT_PATH}: {adata.shape}")
