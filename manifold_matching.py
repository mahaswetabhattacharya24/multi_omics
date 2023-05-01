import os
import sys

import scipy
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from pymatcher import matcher
import pandas as pd
import umap
import scanpy
import anndata
import time
import seaborn as sns

if __name__=="__main__":
    USAGE = "python test_phantom.py <directory>"
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
    #%% Extract scrna features
    dataset_name = 'podocytes_scrna_all'
    adata = scanpy.read_h5ad(filename=''.join([sys.argv[1], dataset_name, '.h5ad']))
    scrna_data_meta = adata.obs
    scrna_data_all = pd.DataFrame(adata.raw.X.todense(), index = adata.raw.obs_names, columns = adata.raw.var['_index'])

    #%% Add metadata to scrna
    scrna_data_meta['DNstage'] = (scrna_data_meta['stim']=='DIAB').astype(int)
    scrna_ctrl_loc = np.where(scrna_data_meta['stim']=='CONTROL')[0]
    scrna_diab_loc = np.where(scrna_data_meta['stim']=='DIAB')[0]

    #%% Extract morphometric features(original data)
    morpho_data_raw = pd.read_excel(''.join([sys.argv[1], 'scRNA_PAS_podocyte_features.xlsx']))
    morpho_data_meta = morpho_data_raw[['DNstage', 'WSIno', 'patchno', 'podno']]
    morpho_data_all = morpho_data_raw.drop(columns = ['DNstage', 'WSIno', 'patchno', 'podno'])

    #%% Lambda values
    morpho_lam = -0.7
    scrna_lam = -2

    morpho_data_temp = pd.read_csv(''.join([sys.argv[1], 'Latent_space/Morpho_allfeats/full_morpho_', str(morpho_lam), '.csv']),header=None)
    scrna_data_temp = pd.read_csv(''.join([sys.argv[1], 'Latent_space/scRNAseq_features/embedding_data_lambda_', str(scrna_lam), '.csv']), index_col=0)
    scrna_data_temp = scrna_data_temp.iloc[:,:-1]

    #%% Labels
    morpho_data_meta.loc[:, 'stim'] = np.where(morpho_data_meta.loc[:, 'DNstage']==1, 'DIAB', 'CONTROL').tolist()
    morpho_ctrl_loc = np.where(morpho_data_meta['DNstage']==0)[0]
    morpho_diab_loc = np.where(morpho_data_meta['DNstage']==1)[0]

    #%% UMAP on the embedding space
    scrna_umap = umap.UMAP(n_neighbors = 30, metric='cosine', min_dist = 0.3,n_components=3).fit(scrna_data_temp)
    morpho_umap = umap.UMAP(n_neighbors = 30, metric='cosine', min_dist = 0.3,n_components=3).fit(morpho_data_temp)# morpho_pca[:,:40] for Seurat like results

    #%% Run MATCHER on the umap embeddings
    scrna_data_matcher = scrna_umap.embedding_
    morpho_data_matcher = morpho_umap.embedding_
    all_datasets = [scrna_data_matcher, morpho_data_matcher]
    m = matcher.MATCHER(all_datasets,doumap=False)
    m.infer()

    #%% Plot
    #% Plot master times for each modality
    n_feats_modal = [morpho_lam, scrna_lam]
    fig_names = ''.join(['UMAP based on embeddings, morphometric lambda 10^: ']+[str(n_feats_modal[0]), ' and scrnaseq lambda 10^ ']+[str(n_feats_modal[1]), '.'])

    fig, axs = plt.subplots(2, 4, figsize = (20,10),constrained_layout = True)
    fig.suptitle(fig_names)

    # Plot of split between control and diab(Tx)
    bg = axs[0,0].scatter(scrna_umap.embedding_[:, 0], scrna_umap.embedding_[:,1],
                        s = np.ones(scrna_umap.embedding_.shape[0])*5, c = scrna_data_meta['DNstage'])
    axs[0,0].set_title('Tx embedding')
    lp = lambda i: axs[0,0].plot([],color=bg.cmap(bg.norm(i)), ms=np.sqrt(81), mec="none",
                            label="{}".format(['CTRL', 'DIAB'][i]), ls="", marker="o")[0]
    handles = [lp(i) for i, e in enumerate(scrna_data_meta['stim'].unique())]
    axs[0,0].legend(handles=handles, loc = 'upper right')

    # Plot of split between control and diab(morpho)
    bg = axs[1,0].scatter(morpho_umap.embedding_[:, 0], morpho_umap.embedding_[:,1],
                        s = np.ones(morpho_umap.embedding_.shape[0])*5, c = morpho_data_meta['DNstage'])
    axs[1,0].set_title('Morphometrics embedding')
    lp = lambda i: axs[1,0].plot([],color=bg.cmap(bg.norm(i)), ms=np.sqrt(81), mec="none",
                            label="{}".format(['CTRL', 'DIAB'][i]), ls="", marker="o")[0]
    handles = [lp(i) for i, e in enumerate(morpho_data_meta['stim'].unique())]
    axs[1,0].legend(handles=handles, loc = 'upper right')

    # Plot of mastertime(all)(Tx)
    axs[0,1].set_xlim(axs[0,0].get_xlim())
    axs[0,1].set_ylim(axs[0,0].get_ylim())
    bg = axs[0,1].scatter(scrna_umap.embedding_[:, 0], scrna_umap.embedding_[:,1],
                        s = np.ones(scrna_umap.embedding_.shape[0])*5,
                        c = m.master_time[0])
    fig.colorbar(bg, ax = axs[0,1])
    axs[0,1].set_title('All Tx cells')

    # Plot of mastertime(ctrl)(Tx)
    axs[0,2].set_xlim(axs[0,0].get_xlim())
    axs[0,2].set_ylim(axs[0,0].get_ylim())
    bg = axs[0,2].scatter(scrna_umap.embedding_[scrna_ctrl_loc, 0], scrna_umap.embedding_[scrna_ctrl_loc,1],
                        s = np.ones(scrna_ctrl_loc.shape[0])*5,
                        c = m.master_time[0][scrna_ctrl_loc,0])
    fig.colorbar(bg, ax = axs[0,2])
    axs[0,2].set_title('CTRL Tx cells')

    # Plot of mastertime(diab)(Tx)
    axs[0,3].set_xlim(axs[0,0].get_xlim())
    axs[0,3].set_ylim(axs[0,0].get_ylim())
    bg = axs[0,3].scatter(scrna_umap.embedding_[scrna_diab_loc, 0], scrna_umap.embedding_[scrna_diab_loc,1],
                        s = np.ones(scrna_diab_loc.shape[0])*5,
                        c = m.master_time[0][scrna_diab_loc,0])
    fig.colorbar(bg, ax = axs[0,3])
    axs[0,3].set_title('DIAB Tx cells')

    # Plot of mastertime(all)(morpho)
    axs[1,1].set_xlim(axs[1,0].get_xlim())
    axs[1,1].set_ylim(axs[1,0].get_ylim())
    bg = axs[1,1].scatter(morpho_umap.embedding_[:, 0], morpho_umap.embedding_[:,1],
                        s = np.ones(morpho_umap.embedding_.shape[0])*5,
                        c = m.master_time[1])
    fig.colorbar(bg, ax = axs[1,1])
    axs[1,1].set_title('All morpho cells')

    # Plot of mastertime(ctrl)(morpho)
    axs[1,2].set_xlim(axs[1,0].get_xlim())
    axs[1,2].set_ylim(axs[1,0].get_ylim())
    bg = axs[1,2].scatter(morpho_umap.embedding_[morpho_ctrl_loc, 0], morpho_umap.embedding_[morpho_ctrl_loc,1],
                        s = np.ones(morpho_ctrl_loc.shape[0])*5,
                        c = m.master_time[1][morpho_ctrl_loc,0])
    fig.colorbar(bg, ax = axs[1,2])
    axs[1,2].set_title('CTRL morpho cells')

    # Plot of mastertime(diab)(morpho)
    axs[1,3].set_xlim(axs[1,0].get_xlim())
    axs[1,3].set_ylim(axs[1,0].get_ylim())
    bg = axs[1,3].scatter(morpho_umap.embedding_[morpho_diab_loc, 0], morpho_umap.embedding_[morpho_diab_loc,1],
                        s = np.ones(morpho_diab_loc.shape[0])*5,
                        c = m.master_time[1][morpho_diab_loc,0])
    fig.colorbar(bg, ax = axs[1,3])
    axs[1,3].set_title('DIAB morpho cells')
    fig.savefig(''.join([sys.argv[1], 'UMAP-matcher-Embeddings_scrna_lambda_', str(n_feats_modal[0]), '_morpho_lambda_', str(n_feats_modal[1]), '.jpg']), dpi= 600)#'lam_', str(lam_val),
    #%% Correlation
    recons_sim = m.recons_sim(0, 1, np.arange(all_datasets[0].shape[1]), np.arange(all_datasets[1].shape[1]))
    recons_sim_scrna_corr = recons_sim[1]
    recons_sim_morpho_corr = recons_sim[0]
    #%% MATCHER mapped plot
    fi=plt.figure()
    ax = fi.add_subplot(111)
    bg = ax.scatter(morpho_umap.embedding_[:, 0], morpho_umap.embedding_[:,1],
                        s = np.ones(morpho_umap.embedding_.shape[0])*5,
                        c = 'gainsboro')
    fg = ax.scatter(recons_sim_scrna_corr[:, 0], recons_sim_scrna_corr[:,1],
                        s = np.ones(recons_sim_scrna_corr.shape[0])*5,
                        c = np.linspace(0, 1, 1000))
    fi.colorbar(fg, ax = ax)
    ax.set_title('Morphometric Embedding UMAP with scrna cells projected')
    fi.savefig(''.join([sys.argv[1], 'UMAP-MATCHER-Mapped_tx_', 'lam_10^', str(scrna_lam),'_on_morphometric_lam_10^',str(morpho_lam),'.jpg']), dpi = 600)
    #%% MATCHER mapped plot  2
    fi=plt.figure()
    ax = fi.add_subplot(111)
    bg = ax.scatter(scrna_umap.embedding_[:, 0], scrna_umap.embedding_[:,1],
                        s = np.ones(scrna_umap.embedding_.shape[0])*5,
                        c = 'gainsboro')
    fg = ax.scatter(recons_sim_morpho_corr[:, 0], recons_sim_morpho_corr[:,1],
                        s = np.ones(recons_sim_morpho_corr.shape[0])*5,
                        c = np.linspace(0, 1, 1000))
    fi.colorbar(fg, ax = ax)
    ax.set_title('scRNA Embedding UMAP with morpho cells projected')
    fi.savefig(''.join([sys.argv[1], 'UMAP-MATCHER-Mapped_morpho_', 'lam_10^', str(morpho_lam),'_on_morphometric_lam_10^',str(scrna_lam),'.jpg']), dpi = 600)