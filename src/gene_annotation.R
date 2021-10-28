## Use R package to change gene id to gene name (only used on SHAP plot)
library(data.table)

PATH_TO_DATA_GENE_NAME = './gene_name_QC.txt'    # path to cleaned data with gene annotation (not gene id) (after quatlity control)
PATH_TO_DATA_GENE_ID = './gene_id_QC.txt'    # path to cleaned data with gene id (not gene name) (after quality control)

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("biomaRt")

library(biomaRt)
input_data <- fread(PATH_TO_DATA_GENE_ID)
ensembl_list <- colnames(input_data)[2:ncol(ensembl_list)]

human <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
gene_coords <- getBM(attributes=c("ensembl_gene_id", "external_gene_name"),
                     filters="ensembl_gene_id", values=ensembl_list, mart=human) 
colnames(input_data)[2:ncol(ensemble_list)] <- gene_coords$external_gene_name[match(colnames(input_data)[2:ncol(ensemble_list)], gene_coords$ensembl_gene_id]
fwrite(input_data, PATH_TO_DATA_GENE_NAME)

##replace ensembl_list with a list of ENSGs that you want to convert to gene names. Then gene_coords becomes a data.frame that’s basically a lookup table or dictionary for the ENSGs with their gene names.
