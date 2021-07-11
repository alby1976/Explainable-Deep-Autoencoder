## R code for Gene Set Enrichment Analysis

library(WebGestaltR)
compress_num = 512

str1 <- '' # first section of file (gene module) name
str2 <- '' # second section of file (gene module) name
str3 <- '' # path to save GSEA results
pathway_num <- 5 # the minimum number of pathways to save

for (i in 1:compress_num){
  skip_to_next <- FALSE
  tryCatch({
    num <- as.character(i)
    file_name <- paste(str1, num, str2, sep="")
    curr_data <- read.csv(file_name, sep="\t", header = FALSE)
    GSEA <- WebGestaltR(interestGene = (curr_data),
                        enrichMethod = "GSEA",
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
  else if (!is.null(GSEA) && nrow(GSEA) >= pathway_num) {
    path_to_save <- paste(str3, str1, num,'.csv', sep = '')
    write.csv(GSEA, path_to_save)
  } else { next}
}
