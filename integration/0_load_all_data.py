import scanpy as sc
import os
import anndata as ad
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
#import sctk as sk
import pandas as pd
import pickle
import scipy.sparse as sp
sc.settings.verbosity = 0
from typing import Dict, Optional
import tables
df = pd.read_csv('/nfs/team298/ls34/disease_atlas/paths_diseaseatlas_includinglesional.csv')
paths=df
cb_outcome = {}
for x in df["Sample"]:
    cb_outcome[x] = []
counter = 0

# 3 parts. Set all to true to run from beginning
load_all_adatas = True
add_core=True
apply_qc=True

ADATA_PATH_CORE="/nfs/team298/ls34/adult_skin/preintegration_data/all_noma2.h5ad" # WAS "/nfs/team298/ls34/adult_skin/preintegration_data/all_noma2.h5ad" BEFORE
# For label transfer. LVL2 IS OPTIONAL
ADATA_PATH_FOR_LABELS="/nfs/team298/ls34/disease_atlas/mrvi/adata_inflamm_scanvi6_HVG8024_EPOCHS10.h5ad.v2"
LVL3_ANNOTATION="lvl3_annotation"
LVL2_ANNOTATION="lvl3_annotation"
LVL0_ANNOTATION="lvl0_annotation"


def _fill_adata_slots_automatically(adata, d):
    """Add other information to the adata object in the appropriate slot."""

    # TODO: what about "features_analyzed_inds"?  If not all features are analyzed, does this work?

    for key, value in d.items():
        try:
            if value is None:
                continue
            value = np.asarray(value)
            if len(value.shape) == 0:
                adata.uns[key] = value
            elif value.shape[0] == adata.shape[0]:
                if (len(value.shape) < 2) or (value.shape[1] < 2):
                    adata.obs[key] = value
                else:
                    adata.obsm[key] = value
            elif value.shape[0] == adata.shape[1]:
                if value.dtype.name.startswith('bytes'):
                    adata.var[key] = value.astype(str)
                else:
                    adata.var[key] = value
            else:
                adata.uns[key] = value
        except Exception:
            print('Unable to load data into AnnData: ', key, value, type(value))
def anndata_from_h5(file: str,
                    analyzed_barcodes_only: bool = True) -> anndata.AnnData:
    """Load an output h5 file into an AnnData object for downstream work.

    Args:
        file: The h5 file
        analyzed_barcodes_only: False to load all barcodes, so that the size of
            the AnnData object will match the size of the input raw count matrix.
            True to load a limited set of barcodes: only those analyzed by the
            algorithm. This allows relevant latent variables to be loaded
            properly into adata.obs and adata.obsm, rather than adata.uns.

    Returns:
        anndata.AnnData: The anndata object, populated with inferred latent variables
            and metadata.

    """

    d = dict_from_h5(file)
    X = sp.csc_matrix((d.pop('data'), d.pop('indices'), d.pop('indptr')),
                      shape=d.pop('shape')).transpose().tocsr()

    # check and see if we have barcode index annotations, and if the file is filtered
    barcode_key = [k for k in d.keys() if (('barcode' in k) and ('ind' in k))]
    if len(barcode_key) > 0:
        max_barcode_ind = d[barcode_key[0]].max()
        filtered_file = (max_barcode_ind >= X.shape[0])
    else:
        filtered_file = True

    if analyzed_barcodes_only:
        if filtered_file:
            # filtered file being read, so we don't need to subset
            print('Assuming we are loading a "filtered" file that contains only cells.')
            pass
        elif 'barcode_indices_for_latents' in d.keys():
            X = X[d['barcode_indices_for_latents'], :]
            d['barcodes'] = d['barcodes'][d['barcode_indices_for_latents']]
        elif 'barcodes_analyzed_inds' in d.keys():
            X = X[d['barcodes_analyzed_inds'], :]
            d['barcodes'] = d['barcodes'][d['barcodes_analyzed_inds']]
        else:
            print('Warning: analyzed_barcodes_only=True, but the key '
                  '"barcodes_analyzed_inds" or "barcode_indices_for_latents" '
                  'is missing from the h5 file. '
                  'Will output all barcodes, and proceed as if '
                  'analyzed_barcodes_only=False')

    # Construct the anndata object.
    adata = anndata.AnnData(X=X,
                            obs={'barcode': d.pop('barcodes').astype(str)},
                            var={'gene_name': (d.pop('gene_names') if 'gene_names' in d.keys()
                                               else d.pop('name')).astype(str)},
                            dtype=X.dtype)
    adata.obs.set_index('barcode', inplace=True)
    adata.var.set_index('gene_name', inplace=True)

    # For CellRanger v2 legacy format, "gene_ids" was called "genes"... rename this
    if 'genes' in d.keys():
        d['id'] = d.pop('genes')

    # For purely aesthetic purposes, rename "id" to "gene_id"
    if 'id' in d.keys():
        d['gene_id'] = d.pop('id')

    # If genomes are empty, try to guess them based on gene_id
    if 'genome' in d.keys():
        if np.array([s.decode() == '' for s in d['genome']]).all():
            if '_' in d['gene_id'][0].decode():
                print('Genome field blank, so attempting to guess genomes based on gene_id prefixes')
                d['genome'] = np.array([s.decode().split('_')[0] for s in d['gene_id']], dtype=str)

    # Add other information to the anndata object in the appropriate slot.
    _fill_adata_slots_automatically(adata, d)

    # Add a special additional field to .var if it exists.
    if 'features_analyzed_inds' in adata.uns.keys():
        adata.var['cellbender_analyzed'] = [True if (i in adata.uns['features_analyzed_inds'])
                                            else False for i in range(adata.shape[1])]
    elif 'features_analyzed_inds' in adata.var.keys():
        adata.var['cellbender_analyzed'] = [True if (i in adata.var['features_analyzed_inds'].values)
                                            else False for i in range(adata.shape[1])]

    if analyzed_barcodes_only:
        for col in adata.obs.columns[adata.obs.columns.str.startswith('barcodes_analyzed')
                                     | adata.obs.columns.str.startswith('barcode_indices')]:
            try:
                del adata.obs[col]
            except Exception:
                pass
    else:
        # Add a special additional field to .obs if all barcodes are included.
        if 'barcodes_analyzed_inds' in adata.uns.keys():
            adata.obs['cellbender_analyzed'] = [True if (i in adata.uns['barcodes_analyzed_inds'])
                                                else False for i in range(adata.shape[0])]
        elif 'barcodes_analyzed_inds' in adata.obs.keys():
            adata.obs['cellbender_analyzed'] = [True if (i in adata.obs['barcodes_analyzed_inds'].values)
                                                else False for i in range(adata.shape[0])]

    return adata

def load_anndata_from_input(input_file: str) -> anndata.AnnData:
    """Load an input file into an AnnData object (used in report generation).
    Equivalent to something like scanpy.read(), but uses cellbender's io.

    Args:
        input_file: The raw data file

    Returns:
        adata.AnnData: The anndata object

    """

    # Load data as dict.
    d = load_data(input_file=input_file)

    # For purely aesthetic purposes, rename slots from the plural to singluar.
    for key in ['gene_id', 'barcode', 'genome', 'feature_type', 'gene_name']:
        if key + 's' in d.keys():
            d[key] = d.pop(key + 's')

    # Create anndata object from dict.
    adata = anndata.AnnData(X=d.pop('matrix'),
                            obs={'barcode': d.pop('barcode').astype(str)},
                            var={'gene_name': d.pop('gene_name').astype(str)},
                            dtype=int)
    adata.obs.set_index('barcode', inplace=True)
    adata.var.set_index('gene_name', inplace=True)

    # Add other information to the anndata object in the appropriate slot.
    _fill_adata_slots_automatically(adata, d)

    return adata

def dict_from_h5(file: str) -> Dict[str, np.ndarray]:
    """Read in everything from an h5 file and put into a dictionary.

    Args:
        file: The h5 file

    Returns:
        Dictionary containing all the information from the h5 file
    """
    d = {}
    with tables.open_file(file) as f:
        # read in everything
        for array in f.walk_nodes("/", "Array"):
            d[array.name] = array.read()
    return d


def anndata_from_h5(file: str,
                    analyzed_barcodes_only: bool = True) -> anndata.AnnData:
    """Load an output h5 file into an AnnData object for downstream work.

    Args:
        file: The h5 file
        analyzed_barcodes_only: False to load all barcodes, so that the size of
            the AnnData object will match the size of the input raw count matrix.
            True to load a limited set of barcodes: only those analyzed by the
            algorithm. This allows relevant latent variables to be loaded
            properly into adata.obs and adata.obsm, rather than adata.uns.

    Returns:
        anndata.AnnData: The anndata object, populated with inferred latent variables
            and metadata.

    """

    d = dict_from_h5(file)
    X = sp.csc_matrix((d.pop('data'), d.pop('indices'), d.pop('indptr')),
                      shape=d.pop('shape')).transpose().tocsr()

    # check and see if we have barcode index annotations, and if the file is filtered
    barcode_key = [k for k in d.keys() if (('barcode' in k) and ('ind' in k))]
    if len(barcode_key) > 0:
        max_barcode_ind = d[barcode_key[0]].max()
        filtered_file = (max_barcode_ind >= X.shape[0])
    else:
        filtered_file = True

    if analyzed_barcodes_only:
        if filtered_file:
            # filtered file being read, so we don't need to subset
            print('Assuming we are loading a "filtered" file that contains only cells.')
            pass
        elif 'barcode_indices_for_latents' in d.keys():
            X = X[d['barcode_indices_for_latents'], :]
            d['barcodes'] = d['barcodes'][d['barcode_indices_for_latents']]
        elif 'barcodes_analyzed_inds' in d.keys():
            X = X[d['barcodes_analyzed_inds'], :]
            d['barcodes'] = d['barcodes'][d['barcodes_analyzed_inds']]
        else:
            print('Warning: analyzed_barcodes_only=True, but the key '
                  '"barcodes_analyzed_inds" or "barcode_indices_for_latents" '
                  'is missing from the h5 file. '
                  'Will output all barcodes, and proceed as if '
                  'analyzed_barcodes_only=False')

    # Construct the anndata object.
    adata = anndata.AnnData(X=X,
                            obs={'barcode': d.pop('barcodes').astype(str)},
                            var={'gene_name': (d.pop('gene_names') if 'gene_names' in d.keys()
                                               else d.pop('name')).astype(str)},
                            dtype=X.dtype)
    adata.obs.set_index('barcode', inplace=True)
    adata.var.set_index('gene_name', inplace=True)

    # For CellRanger v2 legacy format, "gene_ids" was called "genes"... rename this
    if 'genes' in d.keys():
        d['id'] = d.pop('genes')

    # For purely aesthetic purposes, rename "id" to "gene_id"
    if 'id' in d.keys():
        d['gene_id'] = d.pop('id')

    # If genomes are empty, try to guess them based on gene_id
    if 'genome' in d.keys():
        if np.array([s.decode() == '' for s in d['genome']]).all():
            if '_' in d['gene_id'][0].decode():
                print('Genome field blank, so attempting to guess genomes based on gene_id prefixes')
                d['genome'] = np.array([s.decode().split('_')[0] for s in d['gene_id']], dtype=str)

    # Add other information to the anndata object in the appropriate slot.
    _fill_adata_slots_automatically(adata, d)

    # Add a special additional field to .var if it exists.
    if 'features_analyzed_inds' in adata.uns.keys():
        adata.var['cellbender_analyzed'] = [True if (i in adata.uns['features_analyzed_inds'])
                                            else False for i in range(adata.shape[1])]
    elif 'features_analyzed_inds' in adata.var.keys():
        adata.var['cellbender_analyzed'] = [True if (i in adata.var['features_analyzed_inds'].values)
                                            else False for i in range(adata.shape[1])]

    if analyzed_barcodes_only:
        for col in adata.obs.columns[adata.obs.columns.str.startswith('barcodes_analyzed')
                                     | adata.obs.columns.str.startswith('barcode_indices')]:
            try:
                del adata.obs[col]
            except Exception:
                pass
    else:
        # Add a special additional field to .obs if all barcodes are included.
        if 'barcodes_analyzed_inds' in adata.uns.keys():
            adata.obs['cellbender_analyzed'] = [True if (i in adata.uns['barcodes_analyzed_inds'])
                                                else False for i in range(adata.shape[0])]
        elif 'barcodes_analyzed_inds' in adata.obs.keys():
            adata.obs['cellbender_analyzed'] = [True if (i in adata.obs['barcodes_analyzed_inds'].values)
                                                else False for i in range(adata.shape[0])]

    return adata


def load_anndata_from_input_and_output(input_file: str,
                                       output_file: str,
                                       analyzed_barcodes_only: bool = True,
                                       input_layer_key: str = 'cellranger',
                                       retain_input_metadata: bool = False,
                                       gene_expression_encoding_key: str = 'cellbender_embedding',
                                       truth_file: Optional[str] = None) -> anndata.AnnData:
    """Load remove-background output count matrix into an anndata object,
    together with remove-background metadata and the raw input counts.

    Args:
        input_file: Raw h5 file (or other compatible remove-background input)
            used as input for remove-background.
        output_file: Output h5 file created by remove-background (can be
            filtered or not).
        analyzed_barcodes_only: Argument passed to anndata_from_h5().
            False to load all barcodes, so that the size of
            the AnnData object will match the size of the input raw count matrix.
            True to load a limited set of barcodes: only those analyzed by the
            algorithm. This allows relevant latent variables to be loaded
            properly into adata.obs and adata.obsm, rather than adata.uns.
        input_layer_key: Key of the anndata.layer that is created for the raw
            input count matrix.
        retain_input_metadata: In addition to loading the CellBender metadata,
            which happens automatically, set this to True to retain all the
            metadata from the raw input file as well.
        gene_expression_encoding_key: The CellBender gene expression embedding
            will be loaded into adata.obsm[gene_expression_encoding_key]
        truth_file: File containing truth data if this is a simulation

    Return:
        anndata.AnnData: AnnData object with counts before and after remove-background,
            as well as inferred latent variables from remove-background.

    """

    # Load input data.
    adata_raw = load_anndata_from_input(input_file=input_file)

    # Load remove-background output data.
    adata_out = anndata_from_h5(output_file, analyzed_barcodes_only=analyzed_barcodes_only)

    # Subset the raw dataset to the relevant barcodes.
    adata_raw = adata_raw[adata_out.obs.index]

    # TODO: keep the stuff from the raw file too: from obs and var and uns
    # TODO: maybe use _fill_adata_slots_automatically()?  or just copy stuff

    # Put count matrices into 'layers' in anndata for clarity.
    adata_out.layers[input_layer_key] = adata_raw.X.copy()
    adata_out.layers['cellbender'] = adata_out.X.copy()

    # Pre-compute a bit of metadata.
    adata_out.var['n_' + input_layer_key] = \
        np.array(adata_out.layers[input_layer_key].sum(axis=0), dtype=int).squeeze()
    adata_out.var['n_cellbender'] = \
        np.array(adata_out.layers['cellbender'].sum(axis=0), dtype=int).squeeze()
    adata_out.obs['n_' + input_layer_key] = \
        np.array(adata_out.layers[input_layer_key].sum(axis=1), dtype=int).squeeze()
    adata_out.obs['n_cellbender'] = \
        np.array(adata_out.layers['cellbender'].sum(axis=1), dtype=int).squeeze()

    # Load truth data, if present.
    if truth_file is not None:
        adata_truth = anndata_from_h5(truth_file, analyzed_barcodes_only=False)
        adata_truth = adata_truth[adata_out.obs.index]
        adata_out.layers['truth'] = adata_truth.X.copy()
        adata_out.var['n_truth'] = np.array(adata_out.layers['truth'].sum(axis=0), dtype=int).squeeze()
        adata_out.obs['n_truth'] = np.array(adata_out.layers['truth'].sum(axis=1), dtype=int).squeeze()
        for key in adata_truth.obs.keys():
            if key.startswith('truth_'):
                adata_out.obs[key] = adata_truth.obs[key].copy()
        for key in adata_truth.uns.keys():
            if key.startswith('truth_'):
                adata_out.uns[key] = adata_truth.uns[key].copy()
        for key in adata_truth.var.keys():
            if key.startswith('truth_'):
                adata_out.var[key] = adata_truth.var[key].copy()

    # Rename the CellBender encoding of gene expression.
    if analyzed_barcodes_only:
        slot = adata_out.obsm
    else:
        slot = adata_out.uns
    embedding_key = None
    for key in ['gene_expression_encoding', 'latent_gene_encoding']:
        if key in slot.keys():
            embedding_key = key
            break
    if gene_expression_encoding_key != embedding_key:
        slot[gene_expression_encoding_key] = slot[embedding_key].copy()
        del slot[embedding_key]

    return adata_out


if load_all_adatas==True:
    adata_dicts={}
    # Iterate through the DataFrame
    for i, (index, row) in enumerate(df.iterrows()):
        counter += 1
        print(i+1, "/", len(df))
        # Access the metadata for each sample 
        sample_id =row["Sample"]  
        cb_path = row["cellbender_path"]
        filtered_path = row["filtered_path"]
        raw_path = row["raw_path"]
        dataset_id = row["dataset_id"]
        GSE = row["GSE_number"]
        site_status=row["Site_status"]
        
        # Try load cellbender if present
        if os.path.exists(cb_path):
            print("CB path exists")
            try:
                adata_i =anndata_from_h5(cb_path)
                print(adata_i.shape, " - ", cb_outcome[sample_id], sample_id, " from ", dataset_id, GSE, cb_path)
            except:
                print("CB PATH EXISTS BUT ERROR WITH", sample_id, " from ", dataset_id, GSE, cb_path)
                print(raw_path)
                print(cb_path)
                print(" ")
               # cb_outcome[sample_id]="cellbender_path_exists_but_didnt_load"
                try:
                    adata_i = sc.read_10x_mtx(filtered_path)
                    adata_i.layers["cellranger"] = adata_i.X
                 #   cb_outcome[sample_id]="non-cellbender"
                    print(adata_i.shape,sample_id, dataset_id)
                except:
                    print(f"Could not FILTERED load {sample_id} from {dataset_id} - {GSE} AFTER TRYING CELLBENDER")
                    continue
       
        # Load cellranger/starsolo output if not CB
        else:
            try:
                print(f"no cb path. Filtered path: {os.path.exists(filtered_path)}" )
                if not GSE == "GSE184320": ### special case for johanna data as needed to add metadata
                    adata_i = sc.read_10x_mtx(filtered_path)
                elif GSE == "GSE184320": ### special case for johanna data as needed to add metadata
                    adata_i = sc.read_h5ad(filtered_path)
                adata_i.layers["cellranger"] = adata_i.X
             #   cb_outcome[sample_id]="non-cellbender"
                print(adata_i.shape,sample_id, dataset_id)
            except:
                print(f"Could not load {sample_id} from {dataset_id} - {GSE}")
                continue
        print(sample_id, adata_i.shape)
        
        # Tidy obs index + var index
        adata_i.obs["sample_id"] = sample_id
        adata_i.obs["barcode"] = adata_i.obs_names
        adata_i.obs["barcode+sample"] = adata_i.obs["barcode"] + "_" + adata_i.obs["sample_id"] 
        adata_i.obs.index = adata_i.obs["barcode+sample"] 
        adata_i.var["gene_symbol"] = adata_i.var_names
        try:
            adata_i.var_names = adata_i.var["gene_ids"]
        except:
            adata_i.var_names = adata_i.var["gene_id"]
            adata_i.var["gene_ids"]= adata_i.var["gene_id"]
        adata_i.var["ensg_id"] = adata_i.var["gene_ids"]
        
        # Assign metadata to adata object for the sample
        adata_i.obs["barcode"]  = adata_i.obs["barcode"].astype(str)
        adata_i.obs["sample_id"]  = adata_i.obs["sample_id"].astype(str)
        adata_i.obs["dataset_id"]  = dataset_id
        adata_i.obs["GSE"]  = GSE
        adata_i.obs["Site_status"]  = site_status
        adata_i.obs["Patient_status"]=row["Patient_status"]
#        adata_i.obs["DonorID"]=row["DonorID"]
        adata_i.obs["Location"]=row["Location"]
        adata_i.obs["Age"]=row["Age"]
        adata_i.obs["Sex"]=row["Sex"]
        adata_i.obs_names_make_unique()
        
        # Only cells with >50 genes
        sc.pp.filter_cells(adata_i, min_genes=50)

        #adata_i.obs["cellbender_outcome"]= cb_outcome[sample_id]
        
        # Add adata to dict of all adatas
        if sample_id not in adata_dicts:
            adata_dicts[sample_id] = []
        adata_dicts[sample_id] = adata_i
        print("")

        #print(adata_i.shape, " - ", cb_outcome[sample_id], sample_id, " from ", dataset_id, GSE, cb_path)
        
    # Join all individual adatas and save
    adata = ad.concat(adata_dicts.values(), join='outer') # label='sample_id', keys=list(adata_dict.keys()))
    adata.obs["dataset_id"]=adata.obs["dataset_id"].astype(str)
    adata.obs["OriginalAnnotation"]= "Missing"
    adata.obs["Chemistry"]= "Missing"
    adata.write('/nfs/team298/ls34/disease_atlas/data/all_raw.h5ad')
    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Saved! Time: {timestamp} ")
else:
    adata=sc.read_h5ad('/nfs/team298/ls34/disease_atlas/data/all_raw.h5ad')

    

if add_core==True:
    # Load core datasets (internal / were already available from previous work)
    adata_all=sc.read_h5ad(ADATA_PATH_CORE)
    sc.pp.filter_cells(adata_all, min_genes=50)
    
    #adata_all = adata_all[adata_all.obs["Status3"]!="HS_Nonlesional"]
    #adata_all = adata_all[adata_all.obs["Status3"]!="BCC_Lesional"]
    
    # Select only genes present in all samples (as some CTCL data is 10x-flex so not all genes present)
    adata_all.var.index = adata_all.var["ensg_id"] 
    shared_indices = adata_all.var.index
    adata = adata[:, adata.var.index.isin(shared_indices)]
    
    # Merge core with all external adata
    adata = ad.concat([adata, adata_all]) #join='outer', label="core_status", keys=["new_raw", "core"]) 
    
   # Map ensg ids to gene symbols
    with open('/lustre/scratch126/cellgen/team298/ls34/gene_ensgids_dictionaries.pkl', 'rb') as file:
        dictionaries = pickle.load(file)
        gene_dict = dictionaries['gene_dict']
        del(dictionaries)
    adata.var["ensg_id"] = adata.var.index
    adata.var["gene_symbol"] = adata.var.index.map(gene_dict)
    adata.var_names = adata.var["gene_symbol"] 
    del(adata.var["gene_symbol"] )
    try:
        del(adata.obs["cell_probability"] )
    except:
        print("no cell probability column")
    # Assign previous labels to new data
    adata_labelled = sc.read_h5ad(ADATA_PATH_FOR_LABELS)
    label_dict = adata_labelled.obs[LVL3_ANNOTATION].to_dict()
    adata.obs['lvl3_annotation'] = adata.obs.index.map(label_dict).fillna("New/unlabelled/excluded")
    try:
        label_dict = adata_labelled.obs[LVL2_ANNOTATION].to_dict()
        adata.obs['lvl2_annotation'] = adata.obs.index.map(label_dict).fillna("New/unlabelled/excluded")
    except:
        print("no lvl2_annotation")
    #label_dict = adata_labelled.obs['lvl1_annotation'].to_dict()
    #adata.obs['lvl1_annotation'] = adata.obs.index.map(label_dict).fillna("New/unlabelled/excluded")
    label_dict = adata_labelled.obs[LVL0_ANNOTATION].to_dict()
    adata.obs['lvl0_annotation'] = adata.obs.index.map(label_dict).fillna("New/unlabelled/excluded")    
    
    adata.write('/nfs/team298/ls34/disease_atlas/data/all_raw_addedcore.h5ad')
    print("finished -> successful")
else:
    adata=sc.read_h5ad('/nfs/team298/ls34/disease_atlas/data/all_raw_addedcore.h5ad')

def apply_qc_thresholds(adata, MIN_N_GENES, MAX_TOTAL_COUNT, MAX_PCT_MT, label, MIN_TOTAL_COUNT=0,):
    """
    Apply thresholds to generate QC column which says if passed all
    """
    ## Cell cycle gene list
    cc_genes_csv=pd.read_csv("/lustre/scratch126/cellgen/team298/sko_expimap_2023/pan_fetal_cc_genes.csv", names=["ind", "gene_ids"], skiprows=1)
    cc_genes_csv = cc_genes_csv["gene_ids"]
    cc_genes_csv = list(cc_genes_csv)

    # Mark MT/ribo/Hb/cell cycle genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains(("^HB[^(P)]")) 
    #adata.var["hb"] = adata.var_names.str.startswith(("HBA1", "HBA2", "HBB", "HBD","HBM", "HBZ", "HBG1", "HBG2", "HBQ1"))
    adata.var["cc_fetal"] = adata.var_names.isin(cc_genes_csv)

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True, log1p=False) #percent_top=[20],
    
    conditions = [
        (adata.obs['n_genes_by_counts'] < MIN_N_GENES),
        (adata.obs['pct_counts_mt'] > MAX_PCT_MT),
        (adata.obs['total_counts'] > MAX_TOTAL_COUNT),
        (adata.obs['total_counts'] < MIN_TOTAL_COUNT),
        (adata.obs['pct_counts_mt'] <= MAX_PCT_MT) & (adata.obs['n_genes_by_counts'] >= MIN_N_GENES) & 
        (adata.obs['total_counts'] <= MAX_TOTAL_COUNT)  & 
        (adata.obs['total_counts'] >= MIN_TOTAL_COUNT)
    ]
    label_suffix = label.split("_")[-1]
    print(label_suffix)
    pass_name = "Pass_" + label_suffix
    values = ['Low_nFeature', 'High_MT', 'High total count', 'Low total count', pass_name]

    adata.obs[label] = np.select(conditions, values)
    adata.obs[label] = adata.obs[label].astype('category')

    print(adata.obs[label].value_counts())

if apply_qc==True:
    apply_qc_thresholds(adata, MIN_N_GENES=500, MAX_TOTAL_COUNT=300_000, MAX_PCT_MT=20,  MIN_TOTAL_COUNT=2000, label="QC_hi")
    apply_qc_thresholds(adata, MIN_N_GENES=200, MAX_TOTAL_COUNT=300_000, MAX_PCT_MT=15,  MIN_TOTAL_COUNT=1000, label="QC_mid")
    adata = adata[adata.obs["QC_mid"].str.startswith("Pass")]
    adata.write('/nfs/team298/ls34/disease_atlas/data/all_raw_addedcore_filteredmidqc.h5ad')
    print("all done. saved to: /nfs/team298/ls34/disease_atlas/data/all_raw_addedcore_filteredmidqc.h5ad")
