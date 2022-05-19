#!/bin/bash
#SBATCH --mail-user=%uR@ucalgary.ca
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time=2:0:0
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8#SBATCH --job-name=hsa00010-TCGA_BRCA_TPM_Regression-job.slurm
#SBATCH --out=%x-%N-%j.out
#SBATCH --error=%x-%N-%j.error

####### Set environment variables ###############

module load python/anaconda3-2019.10-tensorflowgpu
source $HOME/.bash_profile
conda activate XAI

####### Run script ##############################
echo "python src/AutoEncoder.py {base_name}_AE_Geno data/filter/input/hsa00010-TCGA_BRCA_TPM_Regression/hsa00010-TCGA_BRCA_TPM_Regression.csv data/filter/input/hsa00010-TCGA_BRCA_TPM_Regression/hsa00010-TCGA_BRCA_TPM_Regression_gene_id_QC.csv data/filter/model/hsa00010-TCGA_BRCA_TPM_Regression 64"
python src/AutoEncoder.py hsa00010-TCGA_BRCA_TPM_Regression_AE_Geno data/filter/input/hsa00010-TCGA_BRCA_TPM_Regression/hsa00010-TCGA_BRCA_TPM_Regression.csv data/filter/input/hsa00010-TCGA_BRCA_TPM_Regression/hsa00010-TCGA_BRCA_TPM_Regression_gene_id_QC.csv data/filter/model/hsa00010-TCGA_BRCA_TPM_Regression 64
echo "python src/SHAP_combo.py data/filter/input/hsa00010-TCGA_BRCA_TPM_Regression/hsa00010-TCGA_BRCA_TPM_Regression_gene_name_QC.csv data/filter/input/hsa00010-TCGA_BRCA_TPM_Regression/hsa00010-TCGA_BRCA_TPM_Regression_gene_id_QC.csv data/filter/model/hsa00010-TCGA_BRCA_TPM_Regression data/filter/model/hsa00010-TCGA_BRCA_TPM_Regression/shap/bar data/filter/model/hsa00010-TCGA_BRCA_TPM_Regression/shap/scatter data/filter/model/hsa00010-TCGA_BRCA_TPM_Regression/shap/model"
python src/SHAP_combo.py data/filter/input/hsa00010-TCGA_BRCA_TPM_Regression/hsa00010-TCGA_BRCA_TPM_Regression_gene_name_QC.csv data/filter/input/hsa00010-TCGA_BRCA_TPM_Regression/hsa00010-TCGA_BRCA_TPM_Regression_gene_id_QC.csv data/filter/model/hsa00010-TCGA_BRCA_TPM_Regression data/filter/model/hsa00010-TCGA_BRCA_TPM_Regression/shap/bar data/filter/model/hsa00010-TCGA_BRCA_TPM_Regression/shap/scatter data/filter/model/hsa00010-TCGA_BRCA_TPM_Regression/shap/model

####### Clean up ################################
module unload python/anaconda3-2019.10-tensorflowgpu
