library(data.table)
library(stringr)

tpms <- as.data.frame(fread("TCGA_BRCA_TPM.csv"))
row.names(tpms) <- tpms$V1
tpms$V1 <- NULL

sample_info <- fread("gdc_sample_sheet.2021-10-30.tsv")
sample_info$`File Name` <- str_replace_all(sample_info$`File Name`, "\\.gz", "")
sample_info <- sample_info[endsWith(sample_info$`File Name`, "htseq.counts"),]
sample_info$`File Name` <- str_replace_all(sample_info$`File Name`, "\\.counts", "")

colnames(tpms) <- str_replace_all(colnames(tpms), "Unprocessed/", "")
colnames(tpms) <- sample_info$`Sample ID`[match(colnames(tpms), sample_info$`File Name`)]

sample_info <- sample_info[, c("Sample ID", "Sample Type")]

tpms <- tpms[,!duplicated(colnames(tpms))]
tpms <- data.frame(t(tpms))

tpms$phen <- sample_info$`Sample Type`[match(row.names(tpms), sample_info$`Sample ID`)]
#tpms <- tpms[tpms$phen!="Metastatic",]

fwrite(tpms, "TCGA_BRCA_TPM_Regression.csv", row.names = TRUE)
