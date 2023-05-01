import os
import sys
import scanpy
import numpy as np
import pandas as pd
import anndata
import copy

# Global scanpy settings
scanpy.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
scanpy.logging.print_header()
scanpy.settings.set_figure_params(dpi=80, facecolor='white')

if __name__=="__main__":
    USAGE = "python test_phantom.py <directory>"
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    # Load data
    data = pd.read_csv(os.path.join(sys.argv[1], 'Gene_Count_per_Cell.tsv'),sep='\t', index_col=0).transpose()
    sc_data = anndata.AnnData(X=data)

    scanpy.pl.highest_expr_genes(sc_data, n_top=20, )

    scanpy.pp.filter_cells(sc_data, min_genes=200)
    scanpy.pp.filter_genes(sc_data, min_cells=1)

    sc_data.var['mt'] = sc_data.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
    scanpy.pp.calculate_qc_metrics(sc_data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    scanpy.pl.violin(sc_data, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True)

    scanpy.pl.scatter(sc_data, x='total_counts', y='pct_counts_mt')
    scanpy.pl.scatter(sc_data, x='total_counts', y='n_genes_by_counts')

    sc_data.raw = sc_data

    scanpy.pp.normalize_total(sc_data, target_sum=1e4)
    scanpy.pp.log1p(sc_data)

    scanpy.pp.highly_variable_genes(sc_data, min_mean=0.0125, max_mean=3, min_disp=0.5)
    scanpy.pl.highly_variable_genes(sc_data)

    norm_counts = pd.DataFrame(sc_data.X, index=sc_data.obs.index, columns = sc_data.var.index)
    diag_match = norm_counts.index.str.match('(^[^_]+_)').astype(int)
    norm_counts['Condition'] = np.where(diag_match, 'Diabetic', 'Control')
    norm_counts.to_csv(''.join([outputDir, 'sc_normalizedcounts2.csv']))
    sc_data.obs['Condition'] = np.where(diag_match, 'Diabetic', 'Control')

    adata = copy.deepcopy(sc_data)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    scanpy.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    scanpy.pp.scale(adata, max_value=10)
    scanpy.tl.pca(adata, svd_solver='arpack')
    scanpy.pl.pca(adata, color = 'Condition')
    scanpy.pp.neighbors(adata, n_pcs=50)
    scanpy.tl.umap(adata)
    scanpy.pl.umap(adata, color='Condition')