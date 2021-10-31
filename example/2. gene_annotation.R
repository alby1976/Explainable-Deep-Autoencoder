## Use R package to change gene id to gene name (only used on SHAP plot)
if (!requireNamespace("BiocManager", quietly = TRUE)) #Please update R to be the earliest version, otherwise it could cause error.
  install.packages("BiocManager")

BiocManager::install("biomaRt")

library(biomaRt)
library(data.table)

PATH_TO_DATA_GENE_NAME = 'data_example_QC_name.txt'    # path to cleaned data with gene annotation (gene name) (after quatlity control)
PATH_TO_DATA_GENE_ID = 'data_example_QC.csv'    # path to cleaned data with gene id (after quality control)

input_data <- fread(PATH_TO_DATA_GENE_ID)
ensembl_list <- colnames(input_data)[2:ncol(input_data)]

human <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
gene_coords <- getBM(attributes=c("ensembl_gene_id", "external_gene_name"),
                     filters="ensembl_gene_id", values=ensembl_list, mart=human) 
colnames(input_data)[2:ncol(input_data)] <- gene_coords$external_gene_name[match(colnames(input_data)[2:ncol(input_data)], gene_coords$ensembl_gene_id)]
fwrite(input_data, PATH_TO_DATA_GENE_NAME)
##replace ensembl_list with a list of ENSGs that you want to convert to gene names. Then gene_coords becomes a data.frame that’s basically a lookup table or dictionary for the ENSGs with their gene names.
