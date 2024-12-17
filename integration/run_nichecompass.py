#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %load_ext autoreload
# %autoreload 2


# In[ ]:


import os, sys
import random
import warnings
import logging
from datetime import datetime
# import gdown

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import squidpy as sq
from matplotlib import gridspec
from sklearn.preprocessing import MinMaxScaler
from re import sub


from nichecompass.models import NicheCompass
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                create_new_color_dict,
                                compute_communication_gp_network,
                                visualize_communication_gp_network,
                                extract_gp_dict_from_mebocost_es_interactions,
                                extract_gp_dict_from_nichenet_lrt_interactions,
                                extract_gp_dict_from_omnipath_lr_interactions,
                                filter_and_combine_gp_dict_gps,
                                generate_enriched_gp_info_plots)


# In[ ]:


dryrun=False


# ## Custom code

# In[ ]:


def read_and_qc(sample_name, wtsi, path='rawdata.h5ad'):
    r""" This function reads anndata object.
    It also calculates QC metrics. Modify this function if required by your workflow.
    """
    print(path)
    adata = sc.read_h5ad(path)

    adata.uns['spatial'][sample_name] = adata.uns['spatial'].pop(list(adata.uns['spatial'])[0])
    adata.obs['label'] = list(adata.uns['spatial'])[0]
    adata.obs['WTSI_ID'] = wtsi

    # fix TypeError when read in obsm
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
    # Calculate QC metrics
    from scipy.sparse import csr_matrix
    
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.var['MT'] = [gene.startswith('MT-') for gene in adata.var_names]
    # adata.obs['mt_frac'] = adata[:, adata.var['MT'].tolist()].X.sum(1).A.squeeze()/adata.obs['total_counts']
    
    # add sample name to obs names
    # adata.obs["sample"] = [str(i) for i in adata.obs['sample']]
    # adata.obs_names = adata.obs["sample"] \
    #                       + '_' + adata.obs_names
    adata.obs.index.name = 'spot_id'
    return adata

from sp_plots.spatial_plots import default_color_dict, create_new_color_dict


# ## Define Paramters

# In[4]:
n_svg = 1024
 
### Dataset ###
n_neighbors = 8

dataset = f"XENIUM_svg{n_svg}_neighbors{n_neighbors}_OCT"
ADATA_PATH= ### put path here

species = "human"


reference_batches = [ ### put sample IDs  here - FROM ADATA.OBS.SAMPLE 
] 

spatial_key = "spatial"
mapping_entity_key = "mapping_entity"

### Model ###
# AnnData keys
counts_key = "counts"
adj_key = "spatial_connectivities"
cat_covariates_keys = ["sample"]
gp_names_key = "nichecompass_gp_names"
active_gp_names_key = "nichecompass_active_gp_names"
gp_targets_mask_key = "nichecompass_gp_targets"
gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
gp_sources_mask_key = "nichecompass_gp_sources"
gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
latent_key = "nichecompass_latent"

# Architecture
cat_covariates_embeds_injection = ["gene_expr_decoder"]
cat_covariates_embeds_nums = [len(reference_batches)] ## number samples
cat_covariates_no_edges = [True]
conv_layer_encoder = "gatv2conv" # change to "gatv2conv" if enough compute and memory
active_gp_thresh_ratio = 0.01

# Trainer
n_epochs = 400
n_epochs_all_gps = 25
lr = 0.001
lambda_edge_recon = 500000.
lambda_gene_expr_recon = 300.
lambda_l1_masked = 0. # increase if gene selection desired
lambda_l1_addon = 100.
edge_batch_size = 1024 # increase if more memory available
n_sampled_neighbors = 4
use_cuda_if_available = True
 

### Analysis ###
cell_type_key = "Annotation"
latent_leiden_resolution = 1
latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
sample_key = "sample"
spot_size = 250
differential_gp_test_results_key = "nichecompass_differential_gp_test_results"


# In[ ]:


warnings.filterwarnings("ignore")


# In[ ]:


# Get time of notebook execution for timestamping saved artifacts
now = datetime.now()
current_timestamp = now.strftime("%Y%m%d_%H%M%S")
current_timestamp += dataset  ## Change this for your own project label
current_timestamp


# In[ ]:


handle='/lustre/scratch126/cellgen/team298/ls34/NicheCompass/ref_only' # make sure inside this path, you have the folders gene_annotations and gene_programs with the files


# In[ ]:


# Define paths
ga_data_folder_path = f"{handle}/gene_annotations"
gp_data_folder_path = f"{handle}/gene_programs"
so_data_folder_path = f"{handle}/spatial_omics"
omnipath_lr_network_file_path = f"{gp_data_folder_path}/omnipath_lr_network.csv"
nichenet_lr_network_file_path = f"{gp_data_folder_path}/nichenet_lr_network_v2_{species}.csv"
nichenet_ligand_target_matrix_file_path = f"{gp_data_folder_path}/nichenet_ligand_target_matrix_v2_{species}.csv"
mebocost_enzyme_sensor_interactions_folder_path = f"{gp_data_folder_path}/metabolite_enzyme_sensor_gps"
gene_orthologs_mapping_file_path = f"{ga_data_folder_path}/human_mouse_gene_orthologs.csv"
artifacts_folder_path = f"{handle}/artifacts"
model_folder_path = f"{artifacts_folder_path}/spatial_reference_mapping/{current_timestamp}/model"
figure_folder_path = f"{artifacts_folder_path}/spatial_reference_mapping/{current_timestamp}/figures"


# In[ ]:


if dryrun != True:
    os.makedirs(ga_data_folder_path, exist_ok=True)
    os.makedirs(gp_data_folder_path, exist_ok=True)
    os.makedirs(model_folder_path, exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)
    os.makedirs(so_data_folder_path, exist_ok=True)


# ### This part connects and retrieve information from database, so re-run if unfortunately the server is too busy and you get error

# In[ ]:


# Retrieve OmniPath GPs (source: ligand genes; target: receptor genes)
logging.info("extract_gp_dict_from_omnipath_lr_interactions")
omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
    species=species,
    min_curation_effort=0,
    load_from_disk=False,
    save_to_disk=True,
    lr_network_file_path=omnipath_lr_network_file_path,
    gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
    plot_gp_gene_count_distributions=False,
    gp_gene_count_distributions_save_path=f"{figure_folder_path}" \
                                           "/omnipath_gp_gene_count_distributions.svg")


# In[ ]:


# Display example OmniPath GP
omnipath_gp_names = list(omnipath_gp_dict.keys())
random.shuffle(omnipath_gp_names)
omnipath_gp_name = omnipath_gp_names[0]
print(f"{omnipath_gp_name}: {omnipath_gp_dict[omnipath_gp_name]}")


# In[ ]:


# Retrieve MEBOCOST GPs (source: enzyme genes; target: sensor genes)
logging.info("extract_gp_dict_from_mebocost_es_interactions")
mebocost_gp_dict = extract_gp_dict_from_mebocost_es_interactions(
    dir_path=mebocost_enzyme_sensor_interactions_folder_path,
    species=species,
    plot_gp_gene_count_distributions=False)


# In[ ]:


# Display example MEBOCOST GP
mebocost_gp_names = list(mebocost_gp_dict.keys())
random.shuffle(mebocost_gp_names)
mebocost_gp_name = mebocost_gp_names[0]
print(f"{mebocost_gp_name}: {mebocost_gp_dict[mebocost_gp_name]}")


# In[ ]:


# Retrieve NicheNet GPs (source: ligand genes; target: receptor genes, target genes)
logging.info("extract_gp_dict_from_nichenet_lrt_interactions")
nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
    species=species,
    version="v2",
    keep_target_genes_ratio=1.,
    max_n_target_genes_per_gp=250,
    load_from_disk=False,
    save_to_disk=True,
    lr_network_file_path=nichenet_lr_network_file_path,
    ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
    gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
    plot_gp_gene_count_distributions=False)


# In[ ]:


# Display example NicheNet GP
nichenet_gp_names = list(nichenet_gp_dict.keys())
random.shuffle(nichenet_gp_names)
nichenet_gp_name = nichenet_gp_names[0]
print(f"{nichenet_gp_name}: {nichenet_gp_dict[nichenet_gp_name]}")


# In[ ]:


# Add GPs into one combined dictionary for model training
combined_gp_dict = dict(omnipath_gp_dict)
combined_gp_dict.update(mebocost_gp_dict)
combined_gp_dict.update(nichenet_gp_dict)


# In[ ]:


# Filter and combine GPs to avoid overlaps
logging.info("filter_and_combine_gp_dict_gps")
combined_new_gp_dict = filter_and_combine_gp_dict_gps(
    gp_dict=combined_gp_dict,
    gp_filter_mode="subset",
    combine_overlap_gps=True,
    overlap_thresh_source_genes=0.9,
    overlap_thresh_target_genes=0.9,
    overlap_thresh_genes=0.9)

print("Number of gene programs before filtering and combining: "
      f"{len(combined_gp_dict)}.")
print(f"Number of gene programs after filtering and combining: "
      f"{len(combined_new_gp_dict)}.")


# ## Preparing Anndata

# In[ ]:


adata_vis=sc.read_h5ad(ADATA_PATH)  




# In[ ]:


def select_slide2(adata, s, s_col='sample'):
    """ This function selects the data for one slide from the spatial anndata object.
    :param adata: Anndata object with multiple spatial experiments
    :param s: name of selected experiment
    :param s_col: column in adata.obs listing experiment name for each location
    """
    slide = adata[adata.obs[s_col].isin([s]), :]
#     s_keys = list(slide.uns['spatial'].keys())
#     s_spatial = np.array(s_keys)[[s in k for k in s_keys]][0]
#     slide.uns['spatial'] = {s_spatial: slide.uns['spatial'][s_spatial]}
    return slide
for batch in reference_batches:
    print(f"Processing batch {batch}...")
    print("Loading data...")
    adata_batch = select_slide2(adata_vis, batch)


# In[ ]:


adata_batch_list = []
print("Processing reference batches...")
for batch in reference_batches:
    print(f"Processing batch {batch}...")
    print("Loading data...")
    adata_batch = select_slide2(adata_vis, batch)
    print(f"Size {adata_batch.shape}")
    print("Computing spatial neighborhood graph...\n")
    # Compute (separate) spatial neighborhood graphs
    logging.info("sq.gr.spatial_neighbors")
    #try:
    sq.gr.spatial_neighbors(adata_batch,
                                coord_type="generic",
                                spatial_key=spatial_key,
                                n_neighs=n_neighbors)
    #except:
    #    continue
    print(f"Spatial neighbours done ## {adata_batch.shape}")

    # Make adjacency matrix symmetric
    adata_batch.obsp[adj_key] = (
        adata_batch.obsp[adj_key].maximum(
            adata_batch.obsp[adj_key].T))
    adata_batch_list.append(adata_batch)
print("List made: ...")
for x in adata_batch_list:
    print(x.shape)
adata_reference = ad.concat(adata_batch_list, join="inner")


# In[ ]:


# Combine spatial neighborhood graphs as disconnected components
batch_connectivities = []
len_before_batch = 0
for i in range(len(adata_batch_list)):
    if i == 0: # first batch
        after_batch_connectivities_extension = sp.csr_matrix(
            (adata_batch_list[0].shape[0],
            (adata_reference.shape[0] -
            adata_batch_list[0].shape[0])))
        batch_connectivities.append(sp.hstack(
            (adata_batch_list[0].obsp[adj_key],
            after_batch_connectivities_extension)))
    elif i == (len(adata_batch_list) - 1): # last batch
        before_batch_connectivities_extension = sp.csr_matrix(
            (adata_batch_list[i].shape[0],
            (adata_reference.shape[0] -
            adata_batch_list[i].shape[0])))
        batch_connectivities.append(sp.hstack(
            (before_batch_connectivities_extension,
            adata_batch_list[i].obsp[adj_key])))
    else: # middle batches
        before_batch_connectivities_extension = sp.csr_matrix(
            (adata_batch_list[i].shape[0], len_before_batch))
        after_batch_connectivities_extension = sp.csr_matrix(
            (adata_batch_list[i].shape[0],
            (adata_reference.shape[0] -
            adata_batch_list[i].shape[0] -
            len_before_batch)))
        batch_connectivities.append(sp.hstack(
            (before_batch_connectivities_extension,
            adata_batch_list[i].obsp[adj_key],
            after_batch_connectivities_extension)))
    len_before_batch += adata_batch_list[i].shape[0]
adata_reference.obsp[adj_key] = sp.vstack(batch_connectivities)

adata_reference.obs[mapping_entity_key] = "reference"


# ### Feature selection

# In[ ]:


# Filter spatially variable genes
logging.info("sq.gr.spatial_autocorr")
sq.gr.spatial_autocorr(adata_reference, mode="moran", genes=adata_reference.var_names)
sv_genes = adata_reference.uns["moranI"].index[:n_svg].tolist()
adata_reference.var["spatially_variable"] = adata_reference.var_names.isin(sv_genes)

adata_reference.var["keep_gene"] = adata_reference.var["spatially_variable"]
adata_reference = adata_reference[:, adata_reference.var["keep_gene"] == True]
print(f"Keeping {len(adata_reference.var_names)} spatially variable, highly "
      "variable or gene program relevant genes.")


# In[ ]:


# Add the GP dictionary as binary masks to the adata
logging.info("add_gps_from_gp_dict_to_adata")
add_gps_from_gp_dict_to_adata(
    gp_dict=combined_new_gp_dict,
    adata=adata_reference,
    gp_targets_mask_key=gp_targets_mask_key,
    gp_targets_categories_mask_key=gp_targets_categories_mask_key,
    gp_sources_mask_key=gp_sources_mask_key,
    gp_sources_categories_mask_key=gp_sources_categories_mask_key,
    gp_names_key=gp_names_key,
    min_genes_per_gp=2,
    min_source_genes_per_gp=1,
    min_target_genes_per_gp=1,
    max_genes_per_gp=None,
    max_source_genes_per_gp=None,
    max_target_genes_per_gp=None)


# In[ ]:


# Initialize model
logging.info("NicheCompass")
model = NicheCompass(adata_reference,
                     counts_key=counts_key,
                     adj_key=adj_key,
                     cat_covariates_embeds_injection=cat_covariates_embeds_injection,
                     cat_covariates_keys=cat_covariates_keys,
                     cat_covariates_no_edges=cat_covariates_no_edges,
                     cat_covariates_embeds_nums=cat_covariates_embeds_nums,
                     gp_names_key=gp_names_key,
                     active_gp_names_key=active_gp_names_key,
                     gp_targets_mask_key=gp_targets_mask_key,
                     gp_targets_categories_mask_key=gp_targets_categories_mask_key,
                     gp_sources_mask_key=gp_sources_mask_key,
                     gp_sources_categories_mask_key=gp_sources_categories_mask_key,
                     latent_key=latent_key,
                     conv_layer_encoder=conv_layer_encoder,
                     active_gp_thresh_ratio=active_gp_thresh_ratio)


# In[ ]:


# Train model
logging.info("model.train")
model.train(n_epochs=n_epochs,
            n_epochs_all_gps=n_epochs_all_gps,
            lr=lr,
            lambda_edge_recon=lambda_edge_recon,
            lambda_gene_expr_recon=lambda_gene_expr_recon,
            lambda_l1_masked=lambda_l1_masked,
            edge_batch_size=edge_batch_size,
            n_sampled_neighbors=n_sampled_neighbors,
            use_cuda_if_available=use_cuda_if_available,
            verbose=True)


# In[ ]:


# Compute latent neighbor graph
logging.info("sc.pp.neighbors")
sc.pp.neighbors(model.adata,
                use_rep=latent_key,
                key_added=latent_key)

# Compute UMAP embedding
logging.info("sc.tl.umap")
sc.tl.umap(model.adata,
           neighbors_key=latent_key)


# In[ ]:


# Save trained model
logging.info("model.save")
model.save(dir_path=f"{model_folder_path}/reference",
           overwrite=True,
           save_adata=True,
           adata_file_name="adata.h5ad")


# In[ ]:





# In[ ]:





# # Analysis

# In[ ]:


samples = model.adata.obs[sample_key].unique().tolist()


# In[ ]:


batch_colors = create_new_color_dict(
    adata=model.adata,
    cat_key=cat_covariates_keys[0])


# Create plot of mapping entity annotations in physical and latent space
groups = None
save_fig = True
file_path = f"{figure_folder_path}/" \
            "batches_latent_physical_space.svg"

fig = plt.figure(figsize=(12, 14))
title = fig.suptitle(t=f"NicheCompass Batches " \
                       "in Latent and Physical Space",
                     y=0.96,
                     x=0.55,
                     fontsize=20)
spec1 = gridspec.GridSpec(ncols=1,
                          nrows=2,
                          width_ratios=[1],
                          height_ratios=[3, 2])
spec2 = gridspec.GridSpec(ncols=len(samples),
                          nrows=2,
                          width_ratios=[1] * len(samples),
                          height_ratios=[3, 2])
axs = []
axs.append(fig.add_subplot(spec1[0]))
sc.pl.umap(adata=model.adata,
           color=[mapping_entity_key],
           groups=groups,
           #palette=mapping_entity_colors,
           title=f"Batches in Latent Space",
           ax=axs[0],
           show=False)
for idx, sample in enumerate(samples):
    axs.append(fig.add_subplot(spec2[len(samples) + idx]))
    sc.pl.spatial(adata=model.adata[model.adata.obs[sample_key] == sample],
                  color=[mapping_entity_key],
                  groups=groups,
                  #palette=mapping_entity_colors,
                  spot_size=spot_size,
                  title=f"Batches in Physical Space \n"
                        f"(Sample: {sample})",
                  legend_loc=None,
                  ax=axs[idx+1],
                  show=False)

# Create and position shared legend
handles, labels = axs[0].get_legend_handles_labels()
lgd = fig.legend(handles,
                 labels,
                 loc="center left",
                 bbox_to_anchor=(0.98, 0.5))
axs[0].get_legend().remove()

# Adjust, save and display plot
plt.subplots_adjust(wspace=0.2, hspace=0.25)
if save_fig:
    fig.savefig(file_path,
                bbox_extra_artists=(lgd, title),
                bbox_inches="tight")
plt.show()


# In[ ]:


batch_colors = create_new_color_dict(
    adata=model.adata,
    cat_key=cat_covariates_keys[0])


# In[ ]:


# Create plot of batch annotations in physical and latent space
groups = None
save_fig = True
file_path = f"{figure_folder_path}/" \
            "batches_latent_physical_space.svg"

fig = plt.figure(figsize=(12, 14))
title = fig.suptitle(t=f"NicheCompass Batches " \
                       "in Latent and Physical Space",
                     y=0.96,
                     x=0.55,
                     fontsize=20)
spec1 = gridspec.GridSpec(ncols=1,
                          nrows=2,
                          width_ratios=[1],
                          height_ratios=[3, 2])
spec2 = gridspec.GridSpec(ncols=len(samples),
                          nrows=2,
                          width_ratios=[1] * len(samples),
                          height_ratios=[3, 2])
axs = []
axs.append(fig.add_subplot(spec1[0]))
sc.pl.umap(adata=model.adata,
           color=[cat_covariates_keys[0]],
           groups=groups,
           palette=batch_colors,
           title=f"Batches in Latent Space",
           ax=axs[0],
           show=False)
for idx, sample in enumerate(samples):
    axs.append(fig.add_subplot(spec2[len(samples) + idx]))
    sc.pl.spatial(adata=model.adata[model.adata.obs[sample_key] == sample],
                  color=[cat_covariates_keys[0]],
                  groups=groups,
                  palette=batch_colors,
                  spot_size=spot_size,
                  title=f"Batches in Physical Space \n"
                        f"(Sample: {sample})",
                  legend_loc=None,
                  ax=axs[idx+1],
                  show=False)

# Create and position shared legend
handles, labels = axs[0].get_legend_handles_labels()
lgd = fig.legend(handles,
                 labels,
                 loc="center left",
                 bbox_to_anchor=(0.98, 0.5))
axs[0].get_legend().remove()

# Adjust, save and display plot
plt.subplots_adjust(wspace=0.2, hspace=0.25)
if save_fig:
    fig.savefig(file_path,
                bbox_extra_artists=(lgd, title),
                bbox_inches="tight")
plt.show()


# In[ ]:


cell_type_colors = create_new_color_dict(
    adata=model.adata,
    cat_key=cell_type_key)


# In[ ]:


# Create plot of cell type annotations in physical and latent space
groups = None
save_fig = True
file_path = f"{figure_folder_path}/" \
            "cell_types_latent_physical_space.svg"

fig = plt.figure(figsize=(12, 14))
title = fig.suptitle(t=f"Cell Types " \
                       "in Latent and Physical Space",
                     y=0.96,
                     x=0.55,
                     fontsize=20)
spec1 = gridspec.GridSpec(ncols=1,
                          nrows=2,
                          width_ratios=[1],
                          height_ratios=[3, 2])
spec2 = gridspec.GridSpec(ncols=len(samples),
                          nrows=2,
                          width_ratios=[1] * len(samples),
                          height_ratios=[3, 2])
axs = []
axs.append(fig.add_subplot(spec1[0]))
sc.pl.umap(adata=model.adata,
           color=[cell_type_key],
           groups=groups,palette=cell_type_colors,
           title=f"Cell Types in Latent Space",
           ax=axs[0],
           show=False)
for idx, sample in enumerate(samples):
    axs.append(fig.add_subplot(spec2[len(samples) + idx]))
    sc.pl.spatial(adata=model.adata[model.adata.obs[sample_key] == sample],
                  color=[cell_type_key],
                  groups=groups,
                  palette=cell_type_colors,
                  spot_size=spot_size,
                  title=f"Cell Types in Physical Space \n"
                        f"(Sample: {sample})",
                  legend_loc=None,
                  ax=axs[idx+1],
                  show=False)

# Create and position shared legend
handles, labels = axs[0].get_legend_handles_labels()
lgd = fig.legend(handles,
                 labels,
                 loc="center left",
                 bbox_to_anchor=(0.98, 0.5))
axs[0].get_legend().remove()

# Adjust, save and display plot
plt.subplots_adjust(wspace=0.2, hspace=0.25)
if save_fig:
    fig.savefig(file_path,
                bbox_extra_artists=(lgd, title),
                bbox_inches="tight")
plt.show()


# In[ ]:


# Compute latent Leiden clustering
sc.tl.leiden(adata=model.adata,
            resolution=latent_leiden_resolution,
            key_added=latent_cluster_key,
            neighbors_key=latent_key)


# In[ ]:


latent_cluster_colors = create_new_color_dict(
    adata=model.adata,
    cat_key=latent_cluster_key)


# In[ ]:


# Create plot of latent cluster / niche annotations in physical and latent space
groups = None # set this to a specific cluster for easy visualization, e.g. ["0"]
save_fig = True
file_path = f"{figure_folder_path}/" \
            f"res_{latent_leiden_resolution}_" \
            "niches_latent_physical_space.svg"

fig = plt.figure(figsize=(12, 14))
title = fig.suptitle(t=f"NicheCompass Niches " \
                       "in Latent and Physical Space",
                     y=0.96,
                     x=0.55,
                     fontsize=20)
spec1 = gridspec.GridSpec(ncols=1,
                          nrows=2,
                          width_ratios=[1],
                          height_ratios=[3, 2])
spec2 = gridspec.GridSpec(ncols=len(samples),
                          nrows=2,
                          width_ratios=[1] * len(samples),
                          height_ratios=[3, 2])
axs = []
axs.append(fig.add_subplot(spec1[0]))
sc.pl.umap(adata=model.adata,
           color=[latent_cluster_key],
           groups=groups,
           palette=latent_cluster_colors,
           title=f"Niches in Latent Space",
           ax=axs[0],
           show=False)
for idx, sample in enumerate(samples):
    axs.append(fig.add_subplot(spec2[len(samples) + idx]))
    sc.pl.spatial(adata=model.adata[model.adata.obs[sample_key] == sample],
                  color=[latent_cluster_key],
                  groups=groups,
                  palette=latent_cluster_colors,
                  spot_size=spot_size,
                  title=f"Niches in Physical Space \n"
                        f"(Sample: {sample})",
                  legend_loc=None,
                  ax=axs[idx+1],
                  show=False)

# Create and position shared legend
handles, labels = axs[0].get_legend_handles_labels()
lgd = fig.legend(handles,
                 labels,
                 loc="center left",
                 bbox_to_anchor=(0.98, 0.5))
axs[0].get_legend().remove()

# Adjust, save and display plot
plt.subplots_adjust(wspace=0.2, hspace=0.25)
if save_fig:
    fig.savefig(file_path,
                bbox_extra_artists=(lgd, title),
                bbox_inches="tight")
plt.show()


# In[ ]:


save_fig = True
file_path = f"{figure_folder_path}/" \
            f"res_{latent_leiden_resolution}_" \
            f"niche_composition_batches.svg"

df_counts = (model.adata.obs.groupby([latent_cluster_key, cat_covariates_keys[0]])
             .size().unstack())
df_counts.plot(kind="bar", stacked=True, figsize=(10,10))
legend = plt.legend(bbox_to_anchor=(1, 1), loc="upper left", prop={'size': 10})
legend.set_title("Batch Annotations", prop={'size': 10})
plt.title("Batch Composition of Niches")
plt.xlabel("Niche")
plt.ylabel("Cell Counts")
if save_fig:
    plt.savefig(file_path,
                bbox_extra_artists=(legend,),
                bbox_inches="tight")


# In[ ]:


save_fig = True
file_path = f"{figure_folder_path}/" \
            f"res_{latent_leiden_resolution}_" \
            f"niche_composition_cell_types.svg"

df_counts = (model.adata.obs.groupby([latent_cluster_key, cell_type_key])
             .size().unstack())
df_counts.plot(kind="bar", stacked=True, figsize=(10,10))
legend = plt.legend(bbox_to_anchor=(1, 1), loc="upper left", prop={'size': 10})
legend.set_title("Cell Type Annotations", prop={'size': 10})
plt.title("Cell Type Composition of Niches")
plt.xlabel("Niche")
plt.ylabel("Cell Counts")
if save_fig:
    plt.savefig(file_path,
                bbox_extra_artists=(legend,),
                bbox_inches="tight")


# In[ ]:


# Check number of active GPs
active_gps = model.get_active_gps()
print(f"Number of total gene programs: {len(model.adata.uns[gp_names_key])}.")
print(f"Number of active gene programs: {len(active_gps)}.")


# In[ ]:


# Display example active GPs
gp_summary_df = model.get_gp_summary()
gp_summary_df[gp_summary_df["gp_active"] == True].head()


# In[ ]:


# Set parameters for differential gp testing
selected_cats = ["0"]
comparison_cats = "rest"
title = f"NicheCompass Strongly Enriched Niche GPs"
log_bayes_factor_thresh = 2.3
save_fig = True
file_path = f"{figure_folder_path}/" \
            f"/log_bayes_factor_{log_bayes_factor_thresh}" \
             "_niches_enriched_gps_heatmap.svg"


# In[ ]:


# Run differential gp testing
enriched_gps = model.run_differential_gp_tests(
    cat_key=latent_cluster_key,
    selected_cats=selected_cats,
    comparison_cats=comparison_cats,
    log_bayes_factor_thresh=log_bayes_factor_thresh)
enriched_number = len(enriched_gps)
print(f"# of enriched gps (for 0 vs rest): {enriched_number}")

if enriched_number < 2:
    enriched_gps = model.run_differential_gp_tests(
        cat_key=latent_cluster_key,
        selected_cats=selected_cats[0],
        comparison_cats=comparison_cats,
        log_bayes_factor_thresh=log_bayes_factor_thresh)
    enriched_number = len(enriched_gps)
    print(f"v2 - # of enriched gps (for 0 vs rest): {enriched_number}")
    
if enriched_number < 2:
    enriched_gps = model.run_differential_gp_tests(
        cat_key=latent_cluster_key,
        selected_cats=None,
        comparison_cats=comparison_cats,
        log_bayes_factor_thresh=log_bayes_factor_thresh)
    enriched_number = len(enriched_gps)
    print(f"v3 - # of enriched gps (For none vs rest_: {enriched_number}")


# In[ ]:


# Results are stored in a df in the adata object
model.adata.uns[differential_gp_test_results_key]
#print(f"Columns: ")


# In[ ]:





# In[ ]:


import pickle
with open(f'{model_folder_path}/nichecompass.pkl', 'wb') as file:
    pickle.dump({'gp_summary_df': gp_summary_df,
                 'enriched_gps': enriched_gps,
                # 'network_df': network_df,
                 #'visualize_communication_gp_network': visualize_communication_gp_network,
                 # 'enriched_gp_summary_df': enriched_gp_summary_df,
                },
                file)


# In[ ]:


# Visualize GP activities of enriched GPs across niches
df = model.adata.obs[[latent_cluster_key] + enriched_gps].groupby(latent_cluster_key).mean()


# In[ ]:


import pickle
with open(f'{model_folder_path}/nichecompass.pkl', 'wb') as file:
    pickle.dump({'gp_summary_df': gp_summary_df,
                 'enriched_gps': enriched_gps,
                 'df': df
                # 'network_df': network_df,
                 #'visualize_communication_gp_network': visualize_communication_gp_network,
                 # 'enriched_gp_summary_df': enriched_gp_summary_df,
                },
                file)


# In[ ]:


# Visualize GP activities of enriched GPs across niches
df = model.adata.obs[[latent_cluster_key] + enriched_gps].groupby(latent_cluster_key).mean()

scaler = MinMaxScaler()
normalized_columns = scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_columns, columns=df.columns)
normalized_df.index = df.index

plt.figure(figsize=(16, 8))  # Set the figure size
ax = sns.heatmap(normalized_df,
            cmap='viridis',
            annot=False,
            linewidths=0)
plt.xticks(rotation=45,
           fontsize=8,
           ha="right"
          )
plt.xlabel("Gene Programs", fontsize=16)
plt.savefig(f"{figure_folder_path}/enriched_gps_heatmap.svg",
            bbox_inches="tight")


# In[ ]:


# Store gene program summary of enriched gene programs
save_file = True
file_path = f"{figure_folder_path}/" \
            f"/log_bayes_factor_{log_bayes_factor_thresh}_" \
            "niche_enriched_gps_summary.csv"

gp_summary_cols = ["gp_name",
                   "n_source_genes",
                   "n_non_zero_source_genes",
                   "n_target_genes",
                   "n_non_zero_target_genes",
                   "gp_source_genes",
                   "gp_target_genes",
                   "gp_source_genes_importances",
                   "gp_target_genes_importances"]

enriched_gp_summary_df = gp_summary_df[gp_summary_df["gp_name"].isin(enriched_gps)]
cat_dtype = pd.CategoricalDtype(categories=enriched_gps, ordered=True)
enriched_gp_summary_df.loc[:, "gp_name"] = enriched_gp_summary_df["gp_name"].astype(cat_dtype)
enriched_gp_summary_df = enriched_gp_summary_df.sort_values(by="gp_name")
enriched_gp_summary_df = enriched_gp_summary_df[gp_summary_cols]

if save_file:
    enriched_gp_summary_df.to_csv(f"{file_path}")
else:
    display(enriched_gp_summary_df)


# In[ ]:


plot_label = f"log_bayes_factor_{log_bayes_factor_thresh}_cluster_{selected_cats[0]}_vs_rest"
save_figs = True

generate_enriched_gp_info_plots(
    plot_label=plot_label,
    model=model,
    sample_key=sample_key,
    differential_gp_test_results_key=differential_gp_test_results_key,
    cat_key=latent_cluster_key,
    cat_palette=latent_cluster_colors,
    n_top_enriched_gp_start_idx=0,
    n_top_enriched_gp_end_idx=10,
    feature_spaces=samples, # ["latent"]
    n_top_genes_per_gp=3,
    save_figs=save_figs,
    figure_folder_path=f"{figure_folder_path}/",
    spot_size=spot_size)


# In[ ]:


try:
    gp_name = "CCL19_ligand_receptor_target_gene_GP"
    network_df = compute_communication_gp_network(
        gp_list=[gp_name],
        model=model,
        group_key=latent_cluster_key,
        n_neighbors=n_neighbors)

    visualize_communication_gp_network(
        adata=model.adata,
        network_df=network_df,
        figsize=(16, 7),
        cat_colors=latent_cluster_colors,
        edge_type_colors=["#1f77b4"], 
        cat_key=latent_cluster_key,
        save=True,
        save_path=f"{figure_folder_path}/gp_network_{gp_name}.svg",
        )
except:
    print("didn't work - CCL19")


# In[ ]:


try:
    gp_name = "CCL21_ligand_receptor_target_gene_GP"
    network_df = compute_communication_gp_network(
        gp_list=[gp_name],
        model=model,
        group_key=latent_cluster_key,
        n_neighbors=n_neighbors)

    visualize_communication_gp_network(
        adata=model.adata,
        network_df=network_df,
        figsize=(16, 7),
        cat_colors=latent_cluster_colors,
        edge_type_colors=["#1f77b4"], 
        cat_key=latent_cluster_key,
        save=True,
        save_path=f"{figure_folder_path}/gp_network_{gp_name}.svg",
        )
except:
    print("didn't work - CCL21 GP")


# In[ ]:


try:
    gp_name = "IGFBP7_ligand_receptor_target_gene_GP"
    network_df = compute_communication_gp_network(
        gp_list=[gp_name],
        model=model,
        group_key=latent_cluster_key,
        n_neighbors=n_neighbors)

    visualize_communication_gp_network(
        adata=model.adata,
        network_df=network_df,
        figsize=(16, 7),
        cat_colors=latent_cluster_colors,
        edge_type_colors=["#1f77b4"], 
        cat_key=latent_cluster_key,
        save=True,
        save_path=f"{figure_folder_path}/gp_network_{gp_name}.svg",
        )
except:
    print("didn't work 0 IGFBP7 GP")


# In[ ]:


import pickle
with open(f'{model_folder_path}/nichecompass2.pkl', 'wb') as file:
    pickle.dump({'gp_summary_df': gp_summary_df,
                 'enriched_gps': enriched_gps,
                 'df': df,
                 'network_df': network_df,
                 #'visualize_communication_gp_network': visualize_communication_gp_network,
                  'enriched_gp_summary_df': enriched_gp_summary_df,
                },
                file)


# In[ ]:


# Save model
logging.info("model.save")
model.save(dir_path=f"{model_folder_path}/reference_query",
           overwrite=True,
           save_adata=True,
           adata_file_name="adata_final.h5ad")
