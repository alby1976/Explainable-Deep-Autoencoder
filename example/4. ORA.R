## R code for Over-Representation Analysis

library(WebGestaltR)
compress_num <- 100 # number of files need to be tested (depends on the number of gene module)

gene_module <- './shap/gene_module' # gene module name without number
PATH_TO_SAVE <- './ORA_result' # path to save ORA results
pathway_num <- 10 # the minimum number of pathways to save

for (i in 1:compress_num){
  skip_to_next <- FALSE
  tryCatch({
    num <- as.character(i)
    file_name <- paste0(gene_module, i, ".csv")
    curr_data <- read.csv(file_name, sep="\t", header = FALSE)
    ORA <- WebGestaltR(interestGene = (curr_data$V1),
                        enrichMethod = "ORA",
                        fdrThr = 0.05,
                        minNum = 5,
                        maxNum = 1000,
                        organism = "hsapiens",
                        enrichDatabase="pathway_KEGG",
                        interestGeneType="ensembl_gene_id", referenceSet = "genome",
                        referenceGeneType = "ensembl_gene_id", isOutput = TRUE,
                        projectName = "DiffExpr")}, error = function(e) {skip_to_next <<- TRUE}
  )
  if (skip_to_next) {next}
  else if (!is.null(ORA) && nrow(ORA) >= pathway_num) {
    path_to_save <- paste(PATH_TO_SAVE, i,'.csv', sep = '')
    write.csv(ORA, path_to_save)
  } else { next}
}
