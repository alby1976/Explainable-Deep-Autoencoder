## Use R package to change gene id to gene name (only used on SHAP plot)

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("biomaRt")

library(biomaRt)
human <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
gene_coords <- getBM(attributes=c("ensembl_gene_id", "external_gene_name"),
                     filters="ensembl_gene_id", values=ensembl_list, mart=human) 

write.csv(gene_coords,'gene_coords.csv')

##replace ensembl_list with a list of ENSGs that you want to convert to gene names. Then gene_coords becomes a data.frame that’s basically a lookup table or dictionary for the ENSGs with their gene names.
