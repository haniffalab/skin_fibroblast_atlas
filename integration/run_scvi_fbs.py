from scvi.external import MRVI
import scvi
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from matplotlib.colors import to_hex
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

adata_path='/nfs/team298/ls34/disease_atlas/mrvi/adata_scvi4_removedjunk_v2.h5ad'
adata=sc.read_h5ad(adata_path)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

HVG_BATCH_KEY = "sample_id"
hvg_number = 6000
HVG_BATCH_MINIMUM=80
MAX_EPOCHS=30

hypoxia = ["VEGFA",
"TF",
"SLC2A1-AS1",
"FOXN1",
"VDAC1",
"ASMT",
"PLS3",
"GPI",
"DARS",
"SNAPC1",
"SEC61G",
"GTF2IRD2B",
"SAP30",
"ZMYND8",
"RSBN1",
"BNIP3L",
"GTF2IRD2",
"STC2",
"NARF",
"HK2",
"INHA",
"PCF11",
"CBWD3",
"RAD51-AS1",
"S100P",
"HIF1A",
]

additional_genes_to_exclude = [                             'JUND', 'HSPA1A', 'DNAJB1', 'EEF1A1', 'HSP90AA1', 'FTH1', 'FTL', 'HSPB1', 'XIST', 'VGLL3', "MEG3",
                              "JUNB", "HSPA1B",  "FOSB", "HSP90AA1", "FOS", "DNAJB4", 'HSPA6', 'JUN', "NEAT1", "SOD2", "SOD3", "G0S2", "MYC"] 


original_hvg = str(hvg_number) + "select" + str(HVG_BATCH_MINIMUM)
additional_genes_to_exclude = additional_genes_to_exclude + hypoxia

mask_to_exclude = (
    #adata_hvg.var.cc_fetal | 
    adata.var.hb | 
    adata.var.mt |
   # adata.var.mt2 |
    #adata.var.col |
    adata.var.ribo |
    adata.var.index.isin(additional_genes_to_exclude)
)
mask_to_include = ~mask_to_exclude
adata  = adata[:, mask_to_include]
sc.pp.highly_variable_genes(adata,  
                            n_top_genes=hvg_number, 
                            subset=False,
                            batch_key=HVG_BATCH_KEY,
                            check_values=False,
                           )  
var_genes_all = adata.var.highly_variable
var_genes_batch = adata.var.highly_variable_nbatches > HVG_BATCH_MINIMUM
var_select = adata.var.highly_variable_nbatches >= HVG_BATCH_MINIMUM
var_genes = var_select.index[var_select]
hvg_number = len(var_genes)
print(f"selected {hvg_number} HVGs!")

adata2=sc.read_h5ad(adata_path)
label_dict = adata.var['highly_variable_nbatches'].to_dict()
adata2.var['highly_variable_nbatches'] = adata2.var.index.map(label_dict).fillna(np.nan)
label_dict = adata.var['highly_variable'].to_dict()
adata2.var['highly_variable'] = adata2.var.index.map(label_dict).fillna(False)
adata2.write(adata_path)
print(f"Added HVGs")

adata=adata2.copy()
adata2=0

best_HVG_BATCH_MINIMUM = None
closest_hvg_number = None
closest_difference = float('inf')
for HVG_BATCH_MINIMUM in [10,20,30,40,50, 60,70, 90,100,110, 120,135, 150,160,180, 200,220,250,300]:
    var_genes_batch = adata.var.highly_variable_nbatches > HVG_BATCH_MINIMUM
    var_select = adata.var.highly_variable_nbatches >= HVG_BATCH_MINIMUM
    var_genes = var_select.index[var_select]
    hvg_number = len(var_genes)
    
    difference = abs(hvg_number - 6000)
    
    if difference < closest_difference:
        closest_difference = difference
        closest_hvg_number = hvg_number
        best_HVG_BATCH_MINIMUM = HVG_BATCH_MINIMUM
HVG_BATCH_MINIMUM=best_HVG_BATCH_MINIMUM
hvg_number=closest_hvg_number
CAT_COVS=[]
CAT_COVS_TEMP = [x.replace("_", "").lower() for x in CAT_COVS] 
collapsed_string = "_".join(CAT_COVS_TEMP)
if len(CAT_COVS) == 0:
    model_details= "HVGNUMBER" + str(hvg_number) + "__MINBATCH" + str(HVG_BATCH_MINIMUM) + "__MAXEPOCHS" + str(MAX_EPOCHS) + "__BATCHKEY" + HVG_BATCH_KEY
else:
    model_details= "HVGNUMBER" + str(hvg_number) + "__MINBATCH" + str(HVG_BATCH_MINIMUM) + "__MAXEPOCHS" + str(MAX_EPOCHS) + "__COVS" + collapsed_string
print(f"selected {hvg_number} HVGs!")
var_select = adata.var.highly_variable_nbatches >= HVG_BATCH_MINIMUM
adata = adata[:, var_select]
print(f"{hvg_number} selected -> {adata.shape}")

run_scanvi=False
run_mrvi=False
SCANVI_LABELS_KEY="lvl0_annotation"
SCANVI_UNLABELLED="New/unlabelled/excluded"
adata=adata.copy()
if run_scanvi==True:
    print("RUN SCANVI")
    def run_scvi(adata_hvg,  hvg_number , max_epochs, batch_size_vae, N_LATENT=10, N_LAYERS=1):
        DISPERSION = 'gene'
        try:
            details = "hvg" + str(hvg_number) +   '_'.join(CATEGORICAL_COV) + '_'.join(CONTINUOUS_COV) +  "_maxepochs" + str(max_epochs) + "_nlatent" + str(N_LATENT)+"nlayers" + str(N_LAYERS) + "_BATCHKEY_" + HVG_BATCH_KEY.replace("_", "").lower() 
        except:
            details="missing"
        adata_save_name = 'umap_' + details +"__1"
        print(adata_save_name)
        scvi.model.SCANVI.setup_anndata(adata_hvg, 
                                 batch_key=HVG_BATCH_KEY,
                                  labels_key=SCANVI_LABELS_KEY,
                                        unlabeled_category=SCANVI_UNLABELLED
                                       )
        model = scvi.model.SCANVI(adata_hvg, 
                        dispersion=DISPERSION,
                        n_latent = N_LATENT, 
                        n_layers = N_LAYERS,
                       )
        model.train(accelerator ='gpu', 
                    max_epochs=max_epochs,             
                    early_stopping=True,
                   early_stopping_patience=5,
                   batch_size=batch_size_vae)
        print("model trained")
        latent = model.get_latent_representation() 

        try:
            count=1
            plt.subplots(figsize=(10, 10))
            for key in model.history.keys():
                plt.subplot(4,3,count)
                plt.plot(model.history[key])
                plt.title(key)
                count+=1
            plt.show()    
        except: 
            print("Error with count")
            try:
                print(count)
            except:
                print("can't print count")
        return adata_hvg, model
elif run_scanvi==False:
    if run_mrvi==False:
        print("RUN scvi")
        sample_key = "sample_id"
        def run_scvi(adata_hvg,  hvg_number , max_epochs,  batch_size_vae, N_LATENT=10, N_LAYERS=1):
            DISPERSION = 'gene'
            try:
                details = "hvg" + str(hvg_number) +   '_'.join(CATEGORICAL_COV) + '_'.join(CONTINUOUS_COV) +  "_maxepochs" + str(max_epochs) + "_nlatent" + str(N_LATENT)+"nlayers" + str(N_LAYERS) + "_BATCHKEY_" + HVG_BATCH_KEY.replace("_", "").lower() 
            except:
                details="missingdetails"
            adata_save_name = 'umap_' + details +"__1"
            print(adata_save_name)
            scvi.model.SCVI.setup_anndata(adata_hvg,  
                                            batch_key=HVG_BATCH_KEY, 
                                           )
            model = scvi.model.SCVI(adata_hvg, 
                           )
            model.train(max_epochs=max_epochs,             
                        early_stopping=True,
                        accelerator='gpu',
                       early_stopping_patience=5,  
                       batch_size=batch_size_vae)
            print("model trained")
            return adata_hvg, model
    elif run_mrvi==True:
        print("RUN MRVI")
        sample_key = "sample_id"
        adata=adata[adata.obs["Site_status"]!="Postrx"]
        adata.obs["Site_status"] = adata.obs["Site_status"].cat.remove_unused_categories()
        def run_scvi(adata_hvg,  hvg_number , max_epochs,  batch_size_vae, N_LATENT=10, N_LAYERS=1):
            DISPERSION = 'gene'
            try:
                details = "hvg" + str(hvg_number) +   '_'.join(CATEGORICAL_COV) + '_'.join(CONTINUOUS_COV) +  "_maxepochs" + str(max_epochs) + "_nlatent" + str(N_LATENT)+"nlayers" + str(N_LAYERS) + "_BATCHKEY_" + HVG_BATCH_KEY.replace("_", "").lower() 
            except:
                details="missingdetails"
            adata_save_name = 'umap_' + details +"__1"
            print(adata_save_name)
            MRVI.setup_anndata(adata_hvg, 
                                           sample_key=sample_key,
                                            batch_key="dataset_id"
                                           )
            model = MRVI(adata_hvg, 
                           )
            model.train(max_epochs=max_epochs,             
                        early_stopping=True,
                       early_stopping_patience=5,  
                        accelerator="gpu",
                       batch_size=batch_size_vae)
            print("model trained")
            return adata_hvg, model
adata, model_test = run_scvi(adata, 
                     hvg_number=hvg_number, 
                              max_epochs=MAX_EPOCHS, 
                          batch_size_vae=512,
                                           N_LATENT=30,
                                          N_LAYERS=2 )
print("trained. now load adata")
adata=sc.read_h5ad(adata_path)
if not run_mrvi==True:
    print("mrvi false")
    latent = model_test.get_latent_representation() 
    adata.obsm["X_scvi"] = latent
    adata.layers["counts"]=adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    neighbor=30
    neighbor_id = "neighbor_30"   
    print("start neighbours")
    sc.pp.neighbors(adata, use_rep = 'X_scvi', metric = "euclidean", n_neighbors=neighbor,key_added=neighbor_id)
    print("neighbours done")
    mindist=0.2
    print("start umap")
    sc.tl.umap(adata, min_dist=mindist, neighbors_key =neighbor_id ) 

    print("finished umap")
    leidenres_list = [0.2]
    neighbor_id = 'neighbor_30'
    for leidenres in leidenres_list:
        print("###", leidenres)
        leiden_id = "leiden_res" + str(leidenres) # gayoso 1.2
        sc.tl.leiden(adata, resolution=leidenres, key_added=leiden_id, neighbors_key=neighbor_id)
    print("prep save")
    adata.write(adata_path+".integrated", compression="gzip")

    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Saved! Time: {timestamp}")

else:
    adata=adata[adata.obs["Site_status"]!="Postrx"]
    adata.obs["Site_status"] = adata.obs["Site_status"].cat.remove_unused_categories()

    latent = model_test.get_latent_representation() 
    adata.obsm["X_mrvi_u"] = latent
    latent_z = model_test.get_latent_representation(give_z=True) 
    adata.obsm["X_mrvi_z"] = latent_z
    neighbor=30
    neighbor_id = "neighbor_30_U"   
    sc.pp.neighbors(adata, use_rep = 'X_mrvi_z', metric = "euclidean", n_neighbors=neighbor,key_added="neighbor_30_Z")
    sc.pp.neighbors(adata, use_rep = 'X_mrvi_u', metric = "euclidean", n_neighbors=neighbor,key_added=neighbor_id)
    print("neighbours done")
    sc.pp.neighbors(adata, use_rep = 'X_mrvi_z', metric = "euclidean", n_neighbors=neighbor,key_added="neighbor_30_Z")
    mindist=0.2
    sc.tl.umap(adata, min_dist=mindist, neighbors_key =neighbor_id ) 
    leidenres_list = [0.5]
    for leidenres in leidenres_list:
        print("###", leidenres)
        leiden_id = "leiden_res" + str(leidenres) # gayoso 1.2
        sc.tl.leiden(adata, resolution=leidenres, key_added=leiden_id, neighbors_key=neighbor_id)
    print("prep save")

    dists = model_test.get_local_sample_distances(
    keep_cell=False, groupby="lvl3_annotation", batch_size=32
    )
    print("dists below")
    print(dists)
    try:
        d1 = dists.loc[{"lvl3_annotation_name": "Th"}].initial_clustering
    except:
        print("fail with: dists.loc[{lvl3_annotation_name: Th}].initial_clustering") 
    try:
        print(model_test.sample_info.columns)
    except:
        print("cant print model_test.sample_info.columns")
    try:
        print(model_test.sample_info["Site_status"].value_counts())
    except:
        print("cant print model_test.sample_info[status].value_counts")
    try:
        sample_cov_keys = ["Site_status"]
        model_test.sample_info["Site_status"] = model_test.sample_info["Site_status"].cat.reorder_categories(
            ["Nonlesional", "Lesional"]
        )  
        de_res = model_test.differential_expression(
            sample_cov_keys=sample_cov_keys,
            store_lfc=True,
        )
        print("de_Res done")
    except:
        print("faill sample_imnfo")
    try:
        print(de_res)
    except:
        print("cant print deres")
    try:
        adata.obs["DE_eff_size"] = de_res.effect_size.sel(covariate="Status_Lesional").values
        print("DE_EFF_SIZE added")
    except:
        print("fail Covid_DE_eff_size")
    try:
        print("da_res below")
        print(model_test.differential_abundance)
    except:
        print("could not print model_test.differential_abundance")
    try:
        sample_cov_keys = ["Site_status"]
        da_res = model_test.differential_abundance(sample_cov_keys=sample_cov_keys)
        covid_log_probs = da_res.Status_log_probs.loc[{"Site_status": "Lesional"}]
        healthy_log_probs = da_res.Status_log_probs.loc[{"Site_status": "Nonlesional"}]
        covid_healthy_log_prob_ratio = covid_log_probs - healthy_log_probs
        adata.obs["DA_lfc"] = covid_healthy_log_prob_ratio.values
    except:
        print(f"fail. da_res is {da_rest}")

    adata.write(f'/nfs/team298/ls34/disease_atlas/mrvi/adata_mrvi_all_scvi4.h5ad')
    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Saved! Time: {timestamp} -> /nfs/team298/ls34/disease_atlas/mrvi/adata_mrvi_all.h5ad")    
    model_test.save(f'/nfs/team298/ls34/disease_atlas/mrvi/scvi4_all_MRVIversion_{hvg_number}_new',
                   save_anndata=True) 

