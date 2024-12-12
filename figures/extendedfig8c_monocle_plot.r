
# Purpose: Monocle Trajetory analysis using F2 population as root node]

##### 1. Load Libraries #####
# Load libraries
library(monocle3)
library(ggplot2)
library(patchwork)



##### 2. Custom Functions ##### 

# Custom function to create a cell_data_set (CDS) object 

## mtx: a count matrix or numeric matrix of gene expression values having genes as rows, and cells as columns.
## cellmeta: a dataframe having cells/barcodes metadata (equivalent to adata.obs)
## genemeta: a dataframe having gene/features metadata (equivalent to adata.vars)

## Notes: 

#   - If the input matrix is in AnnData format (cells as rows, genes as columns),
#     set transpose_matrix = TRUE (default).
#   - If the input matrix is already in Monocle format (genes as rows, cells as columns),
#     set transpose_matrix = FALSE.



create_cds <- function(mtx, cellmeta, genemeta, transpose_matrix = TRUE) {
  # Check if the file exists
  if (!file.exists(mtx)) {
    stop("Error: input matrix file path does not exist.")
  }
  
  # Load the expression matrix
  expression_mtx <- tryCatch(
    readMM(mtx),
    error = function(e) {
      stop("Error: Failed to load the matrix. file not in the correct Matrix Market format.\n", e)
    }
  )
  
  # Optionally transpose the matrix
  if (transpose_matrix) {
    expression_mtx <- t(expression_mtx)
  } else {
    cat("Skipping transpose: assuming matrix in genes x cells format.\n")
  }
  
  # Set the row names and column names of the expression matrix
  rownames(expression_mtx) <- rownames(genemeta)  # Gene names as row names
  colnames(expression_mtx) <- rownames(cellmeta)  # Cell IDs as column names
  
  # Create the CDS object
  
  cat("Creating CellDataSet object.\n")
  cds <- tryCatch(
    new_cell_data_set(
      expression_data = expression_mtx,
      cell_metadata = cellmeta,
      gene_metadata = genemeta
    ),
    error = function(e) {
      stop("Error: Failed to create the CellDataSet object.check all input files are in correct format.\n", e)
    }
  )
  
  cat("CellDataSet object successfully created!\n")
  return(cds)
}

  

##### 3. Main Workflow #####

setwd('~/kc28/monocle3/')


# Step 1: Load Data

# Load cell and gene metadata
cellmeta <- read.csv('/nfs/users/nfs_k/kc28/kc28/monocle3/cell_metadata_HVGS_counts.csv', row.names = 1)
genemeta <- read.csv("/nfs/users/nfs_k/kc28/kc28/monocle3/gene_metadata_HVGS_counts.csv", row.names = 1)


# Step 2: Create cell_data_set (CDS) object 


mtx = "expression_matrix_HVGS_counts.mtx"

cds_HGVS_counts = create_cds(mtx, cellmeta, genemeta, transpose_matrix = TRUE)

# Step 3: Preprocess Data

cds_HGVS_counts <- preprocess_cds( cds_HGVS_counts, method = "PCA", num_dim = 50)
  

# Step 4: Perform Alignemnt - Remove batch effects

cds_HGVS_counts <- align_cds(cds_HGVS_counts, num_dim = 100, alignment_group = "dataset_id", alignment_k = 20, verbose = TRUE)


# Step 5: Dimensionality Reduction using UMAP

cds_HGVS_counts <- reduce_dimension(cds_HGVS_counts, reduction_method='UMAP', verbose = TRUE)


# Step 6: Clustering of cells

cds_HGVS_counts <- cluster_cells(cds_HGVS_counts, reduction_method='UMAP', resolution=1e-4, verbose = TRUE)

# Step 7: Learning the trajectory graph

cds_HGVS_counts <- learn_graph(cds_HGVS_counts, use_partition = TRUE, close_loop = FALSE,
  learn_graph_control = list(
    geodesic_distance_ratio = 0.5,  # Lower for large datasets ; 0.2 - 10, 0.5 - 10 , 0.5 - 8 , 0.2 - 0.8
    minimal_branch_len = 10, # Higher to reduce short branches
    nn.cores=10 # 2/3 of available cores
  ),
  verbose = TRUE)

#saveRDS(cds_HGVS_counts , 'final_cds_HGVS_counts_aligned_graph_learned_nov19.rds')

# Step 8: Order the cells in pseudotime

# F2: Universal as root node - Manually selected the nodes belonging to F2 populations according to the 'test13' annotations. order_cells() launches a graphical user interface for selecting the root nodes.

cds_HGVS_counts = order_cells(cds_HGVS_counts, reduction_method = 'UMAP')


### 4. Plots  ####

custom_colors <- c(
  'F1: Epithelium-associated' = rgb(1, 1, 0.898),
  'F1: Regenerative' = rgb(0.996, 0.809, 0.396),
  'F2: Universal' = rgb(0.814, 0.884, 0.95),
  'F2/3: Bridge' = rgb(0.473, 0.712, 0.851),
  'F3: FRC-like' = rgb(0.997, 0.896, 0.849),
  'F6: Myofibroblast inflammatory' = rgb(0, 1, 1),
  'F6: Myofibroblast' = rgb(0.333, 0.667, 1)
)


p1 = plot_cells(cds_HGVS_counts, color_cells_by="test13", label_cell_groups=FALSE,  label_groups_by_cluster=TRUE,
                label_leaves=FALSE, label_branch_points=FALSE, group_label_size = 2, cell_size = 0.80, trajectory_graph_color = "gray50",  label_principal_points = FALSE ) +  
  scale_color_manual(values = custom_colors) + 
  theme_void() + theme(legend.position = "none")

p2 = plot_cells(cds_HGVS_counts, color_cells_by="pseudotime",
                label_leaves=FALSE,
                label_branch_points=FALSE, trajectory_graph_color = "gray50",
                label_cell_groups=FALSE, cell_size = 0.80) + scale_color_gradientn(colors = c("blue", "grey", "red")) + 
  labs(color = "pseudotime") +  theme_void()

plot1 =  p1 + p2
plot1

png('F2_dataset_cds_F2_as_root_node_new.png', height = 800 , width = 1600)
print(plot1)
dev.off()


saveRDS(cds_HGVS_counts, 'F2_dataset_cds_ordered_object.rds')



#### 5. Record Session Information #######

cat("Recording session information...\n")
session_info <- sessionInfo()
writeLines(capture.output(session_info), "session_info.txt")
