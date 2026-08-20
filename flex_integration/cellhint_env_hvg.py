import scanpy as sc
import cellhint
import re
import pandas as pd
import numpy as np
import pandas as pd


file_path='/data/scRNA/To_Mehdi/Emilia/flex/Integration_mouse_human.h5ad'
adata=sc.read_h5ad(file_path)

sc.pp.normalize_total(adata, target_sum = 1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, batch_key = 'Dataset', subset = True)
sc.pp.scale(adata, max_value = 10)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

adata.obsm['X_pca'] = np.load('/data/scRNA/To_Mehdi/Emilia/flex/pilot2_embeddings/z_laten.npy')

import cellhint
import os
import pandas as pd
from itertools import product

outdir = "/data/scRNA/To_Mehdi/Emilia/flex/pilot2_embeddings/cell_hint/"
os.makedirs(outdir, exist_ok=True)

maximum_novel_options = [0.001, 0.003, 0.005, 0.01]

minimum_unique_options = [
    [0.03, 0.06, 0.10],
    [0.05, 0.10, 0.15],
    [0.075, 0.15, 0.20],
    [0.10, 0.20, 0.30],
    [0.15, 0.25, 0.35],
]

minimum_divide_options = [
    [0.005, 0.01, 0.02],
    [0.01, 0.02, 0.05],
    [0.015, 0.03, 0.075],
    [0.02, 0.05, 0.10],
    [0.03, 0.075, 0.15],
]

results = []

for max_novel, min_unique, min_divide in product(
    maximum_novel_options,
    minimum_unique_options,
    minimum_divide_options
):
    name = (
        f"novel_{max_novel}_"
        f"unique_{'-'.join(map(str, min_unique))}_"
        f"divide_{'-'.join(map(str, min_divide))}"
    )

    print("Running:", name)

    alignment = cellhint.harmonize(
        adata,
        dataset="Dataset",
        cell_type="cell_subtype",
        maximum_novel_percent=max_novel,
        minimum_unique_percents=min_unique,
        minimum_divide_percents=min_divide,
        dataset_order=["human", "Mouse"]
    )

    save_path = f"{outdir}/all_data_alignment_{name}.pkl"
    alignment.write(save_path)

    

 
