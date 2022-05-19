#!/bin/bash
#SBATCH --mail-user=%u@ucalgary.ca
#SBATCH --mail-type=all
#SBATCH --time=0-08:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATcH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --job-name=full_cancer_TCGA_TPM_Regression-job.slurm
#SBATCH --out=%x-%N-%j.out
#SBATCH --error=%x-%N-%j.error
#SBATCH --partition=gpu-v100

####### Set environment variables ###############
module load python/anaconda3-2019.10-tensorflowgpu
source $HOME/.bash_profile
conda activate XAI
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

####### Run script ##############################
echo "python src/AutoEncoder.py -v -sl 32 --fold -s --name=TCGA_BRCA_TPM_Regression_AE_Geno --data=data/TCGA_BRCA_TPM_Regression.csv --transformed_data=data/TCGA_BRCA_TPM_Regression_gene_id_QC.csv --save_dir=data/model/TCGA_BRCA_TPM_Regression/TCGA_BRCA_TPM_Regression  --tune --filter_str 'Primary Tumor' 'Metastatic' --num_workers=8 --val_split=0.2 --patience=10 &> breast-2-log-median-wandb-column-name-log.txt"
srun --ntasks=1 --nodes=1 --gres=gpu:1 --exclusive python src/AutoEncoder.py -v \
-sl 32 \
--fold \
-s \
-b shap/bar \
--save_scatter shap/scatter \
-m shap/gene_model \
-tr 0.2 \
--name=TCGA_BRCA_TPM_Regression_AE_Geno \
--data=data/TCGA_BRCA_TPM_Regression.csv \
--transformed_data=data/TCGA_BRCA_TPM_Regression_gene_id_QC.csv \
--save_dir=data/model/TCGA_BRCA_TPM_Regression/TCGA_BRCA_TPM_Regression \
--tune --filter_str 'Primary Tumor' 'Metastatic' --num_workers=8 \
--val_split=0.2 --patience=10 &> \
breast-2-log-median-wandb-column-name-log.txt &

echo "python src/AutoEncoder.py -v -sl 32 --fold -s --name=TCGA_COAD_TPM_Regression_AE_Geno --data=data/TCGA_COAD_TPM_Regression.csv --transformed_data=data/TCGA_COAD_TPM_Regression_gene_id_QC.csv --save_dir=data/model/TCGA_COAD_TPM_Regression/TCGA_COAD_TPM_Regression  --tune --filter_str 'Primary Tumor' 'Metastatic' --num_workers=8 --val_split=0.2 --patience=10 &> colon-2-log-median-wandb-column-name-log.txt"
srun --ntasks=1 --nodes=1 --gres=gpu:1 --exclusive python src/AutoEncoder.py -v \
-sl 32 \
--fold \
-s \
-b shap/bar \
--save_scatter shap/scatter \
-m shap/gene_model \
-tr 0.2 \
--name=TCGA_COAD_TPM_Regression_AE_Geno \
--data=data/TCGA_COAD_TPM_Regression.csv \
--transformed_data=data/TCGA_COAD_TPM_Regression_gene_id_QC.csv \
--save_dir=data/model/TCGA_COAD_TPM_Regression/TCGA_COAD_TPM_Regression \
--tune --filter_str 'Primary Tumor' 'Metastatic' --num_workers=8 \
--val_split=0.2 --patience=10 &> \
colon-2-log-median-wandb-column-name-log.txt &

echo "python src/AutoEncoder.py -v -sl 32 --fold -s --name=TCGA_KIRC_TPM_Regression_AE_Geno --data=data/TCGA_KIRC_TPM_Regression.csv --transformed_data=data/TCGA_KIRC_TPM_Regression_gene_id_QC.csv --save_dir=data/model/TCGA_KIRC_TPM_Regression/TCGA_KIRC_TPM_Regression  --tune --filter_str 'Primary Tumor' 'Metastatic' --num_workers=8 --val_split=0.2 --patience=10 &> renal-2-log-median-wandb-column-name-log.txt"
srun --ntasks=1 --nodes=1 --gres=gpu:1 --exclusive python src/AutoEncoder.py -v \
-sl 32 \
--fold \
-s \
-b shap/bar \
--save_scatter shap/scatter \
-m shap/gene_model \
-tr 0.2 \
--name=TCGA_KIRC_TPM_Regression_AE_Geno \
--data=data/TCGA_KIRC_TPM_Regression.csv \
--transformed_data=data/TCGA_KIRC_TPM_Regression_gene_id_QC.csv \
--save_dir=data/model/TCGA_KIRC_TPM_Regression/TCGA_KIRC_TPM_Regression \
--tune --filter_str 'Primary Tumor' 'Metastatic' --num_workers=8 \
--val_split=0.2 --patience=10 &> \
renal-2-log-median-wandb-column-name-log.txt &

echo "python src/AutoEncoder.py -v -sl 32 --fold -s --name=TCGA_LUAD_TPM_Regression_AE_Geno --data=data/TCGA_LUAD_TPM_Regression.csv --transformed_data=data/TCGA_LUAD_TPM_Regression_gene_id_QC.csv --save_dir=data/model/TCGA_LUAD_TPM_Regression/TCGA_LUAD_TPM_Regression  --tune --filter_str 'Primary Tumor' 'Metastatic' --num_workers=8 --val_split=0.2 --patience=10 &> lung-2-log-median-wandb-column-name-log.txt"
srun --ntasks=1 --nodes=1 --gres=gpu:1 --exclusive python src/AutoEncoder.py -v \
-sl 32 \
--fold \
-s \
-b shap/bar \
--save_scatter shap/scatter \
-m shap/gene_model \
-tr 0.2 \
--name=TCGA_LUAD_TPM_Regression_AE_Geno \
--data=data/TCGA_LUAD_TPM_Regression.csv \
--transformed_data=data/TCGA_LUAD_TPM_Regression_gene_id_QC.csv \
--save_dir=data/model/TCGA_LUAD_TPM_Regression/TCGA_LUAD_TPM_Regression \
--tune --filter_str 'Primary Tumor' 'Metastatic' --num_workers=8 \
--val_split=0.2 --patience=10 &> \
lung-2-log-median-wandb-column-name-log.txt 

####### Clean up ################################
module unload python/anaconda3-2019.10-tensorflowgpu

