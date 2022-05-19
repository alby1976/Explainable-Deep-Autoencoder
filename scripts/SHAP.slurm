#!/bin/bash
#SBATCH --mail-user=%u@ucalgary.ca
#SBATCH --mail-type=all
#SBATCH --time=0-08:00:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATcH --mem=32GB
#SBATCH --cpus-per-task=16
#SBATCH --job-name=XAE4Exp-SHAP-job.slurm
#SBATCH --out=%x-%N-%j.out
#SBATCH --error=%x-%N-%j.error
#SBATCH --partition=cpu2021,single,lattice,parallel,cpu2013,cpu2019
#SBATCH --time=7-0:0:0

####### Set environment variables ###############
echo "made it"
source $HOME/.bash_profile
echo "made it 2"
conda activate XAI
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

####### Run script ##############################
echo "...Calculating SHAP values for Breast cancer..."
srun -n 1 --nodes=1  --exclusive \
python src/SHAP_combo.py -s \
 --model_name SHAP_TCGA_BRCA_TPM_Regression \
 -name data/TCGA_BRCA_TPM_Regression_gene_name.csv \
 -id data/TCGA_BRCA_TPM_Regression_gene_id_QC.csv \
 --ae_result data/model/TCGA_BRCA_TPM_Regression/TCGA_BRCA_TPM_Regression/TCGA_BRCA_TPM_Regression_AE_Geno-output.csv \
 --col_mask data/TCGA_BRCA_TPM_Regression_column_mask.csv \
 -b data/model/TCGA_BRCA_TPM_Regression/TCGA_BRCA_TPM_Regression/shap/bar \
 --save_scatter data/model/TCGA_BRCA_TPM_Regression/TCGA_BRCA_TPM_Regression/shap/scatter \
 -m data/model/TCGA_BRCA_TPM_Regression/TCGA_BRCA_TPM_Regression/shap/gene_model \
 -w 8 &> breast-shap.txt &

echo "...Calculating SHAP values for Colon cancer..."
srun -n 1 --nodes=1  --exclusive \
python src/SHAP_combo.py -s \
 --model_name SHAP_TCGA_COAD_TPM_Regression \
 -name data/TCGA_COAD_TPM_Regression_gene_name.csv \
 -id data/TCGA_COAD_TPM_Regression_gene_id_QC.csv \
 --ae_result data/model/TCGA_COAD_TPM_Regression/TCGA_COAD_TPM_Regression/TCGA_COAD_TPM_Regression_AE_Geno-output.csv \
 --col_mask data/TCGA_COAD_TPM_Regression_column_mask.csv \
 -b data/model/TCGA_COAD_TPM_Regression/TCGA_COAD_TPM_Regression/shap/bar \
 --save_scatter data/model/TCGA_COAD_TPM_Regression/TCGA_COAD_TPM_Regression/shap/scatter \
 -m data/model/TCGA_COAD_TPM_Regression/TCGA_COAD_TPM_Regression/shap/gene_model \
 -w 8 &> colon-shap.txt &

echo "...Calculating SHAP values for Kidney cancer..."
srun -n 1 --nodes=1  --exclusive \
python src/SHAP_combo.py -s \
 --model_name SHAP_TCGA_KIRC_TPM_Regression \
 -name data/TCGA_KIRC_TPM_Regression_gene_name.csv \
 -id data/TCGA_KIRC_TPM_Regression_gene_id_QC.csv \
 --ae_result data/model/TCGA_KIRC_TPM_Regression/TCGA_KIRC_TPM_Regression/TCGA_KIRC_TPM_Regression_AE_Geno-output.csv \
 --col_mask data/TCGA_KIRC_TPM_Regression_column_mask.csv \
 -b data/model/TCGA_KIRC_TPM_Regression/TCGA_KIRC_TPM_Regression/shap/bar \
 --save_scatter data/model/TCGA_KIRC_TPM_Regression/TCGA_KIRC_TPM_Regression/shap/scatter \
 -m data/model/TCGA_KIRC_TPM_Regression/TCGA_KIRC_TPM_Regression/shap/gene_model \
 -w 8 &> kidney-shap.txt &

echo "...Calculating SHAP values for Lung cancer..."
srun -n 1 --nodes=1  --exclusive \
python src/SHAP_combo.py -s \
 --model_name SHAP_TCGA_LUAD_TPM_Regression \
 -name data/TCGA_LUAD_TPM_Regression_gene_name.csv \
 -id data/TCGA_LUAD_TPM_Regression_gene_id_QC.csv \
 --ae_result data/model/TCGA_LUAD_TPM_Regression/TCGA_LUAD_TPM_Regression/TCGA_LUAD_TPM_Regression_AE_Geno-output.csv \
 --col_mask data/TCGA_LUAD_TPM_Regression_column_mask.csv \
 -b data/model/TCGA_LUAD_TPM_Regression/TCGA_LUAD_TPM_Regression/shap/bar \
 --save_scatter data/model/TCGA_LUAD_TPM_Regression/TCGA_LUAD_TPM_Regression/shap/scatter \
 -m data/model/TCGA_LUAD_TPM_Regression/TCGA_LUAD_TPM_Regression/shap/gene_model \
 -w 8 &> lung-shap.txt &

echo "...Calculating SHAP values for Prostate cancer..."
srun -n 1 --nodes=1  --exclusive \
python src/SHAP_combo.py -s \
 --model_name SHAP_TCGA_PRAD_TPM_Regression \
 -name data/TCGA_PRAD_TPM_Regression_gene_name.csv \
 -id data/TCGA_PRAD_TPM_Regression_gene_id_QC.csv \
 --ae_result data/model/TCGA_PRAD_TPM_Regression/TCGA_PRAD_TPM_Regression/TCGA_PRAD_TPM_Regression_AE_Geno-output.csv \
 --col_mask data/TCGA_PRAD_TPM_Regression_column_mask.csv \
 -b data/model/TCGA_PRAD_TPM_Regression/TCGA_PRAD_TPM_Regression/shap/bar \
 --save_scatter data/model/TCGA_PRAD_TPM_Regression/TCGA_PRAD_TPM_Regression/shap/scatter \
 -m data/model/TCGA_PRAD_TPM_Regression/TCGA_PRAD_TPM_Regression/shap/gene_model \
 -w 8 &> prostate-shap.txt &

echo "...Caculating SHAP values for Thyroid cancer..."
srun -n 1 --nodes=1  --exclusive \
python src/SHAP_combo.py -s \
 --model_name SHAP_TCGA_THCA_TPM_Regression \
 -name data/TCGA_THCA_TPM_Regression_gene_name.csv \
 -id data/TCGA_THCA_TPM_Regression_gene_id_QC.csv \
 --ae_result data/model/TCGA_THCA_TPM_Regression/TCGA_THCA_TPM_Regression/TCGA_THCA_TPM_Regression_AE_Geno-output.csv \
 --col_mask data/TCGA_THCA_TPM_Regression_column_mask.csv \
 -b data/model/TCGA_THCA_TPM_Regression/TCGA_THCA_TPM_Regression/shap/bar \
 --save_scatter data/model/TCGA_THCA_TPM_Regression/TCGA_THCA_TPM_Regression/shap/scatter \
 -m data/model/TCGA_THCA_TPM_Regression/TCGA_THCA_TPM_Regression/shap/gene_model \
 -w 8 &> thyroid-shap.txt &

####### Clean up #################################

