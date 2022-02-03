# Filters Dataset by KEGG Pathway and uses PyEnsembl to get EnsemblID
from pyensembl import EnsemblRelease
from pathlib import Path
import pathlib
import pandas as pd
import numpy as np
import subprocess
import sys
import os


def get_gene_ids_from_string(ensembl_release: int, genes: str) -> np.ndarray:
    return get_gene_ids(ensembl_release=ensembl_release, gene_list=genes.split(';'))


def get_gene_ids(ensembl_release: int, gene_list: np.ndarray) -> np.ndarray:
    gene_data = EnsemblRelease(release=ensembl_release, species='human', server='ftp://ftp.ensembl.org/')
    gene_data.download()
    gene_data.index()
    ids = []
    for gene in gene_list:
        try:
            ids.append(gene_data.gene_ids_of_gene_name(gene_name=gene)[0])
        except ValueError:
            ids.append(gene)
    return np.array(ids)


def mkdir_p(directory: pathlib.Path):
    """make a directory (directory) if it doesn't exist"""
    directory.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('less than 6 command line arguments')
        print('python GeneSelectionPathway.py ensemble_version dir_original_data '
              'filename_pathway_data dir_filtered_data dir_AE_model session_id')
        sys.exit(-1)

    # data setup
    ensembl_version = int(sys.argv[1])  # Ensembl Release version e.g. 104
    path_to_original_data = Path(sys.argv[2])  # path to original data e.g. './data/input/'
    pathway_data = Path(sys.argv[3])  # pathway data e.g. './data/pathway.csv'
    path_to_save_filtered_data = Path(sys.argv[4])  # base dir to saved filtered original data e.g. './data/filter'
    save_dir = Path(sys.argv[5])  # base directory to save AE models e.g. '.data/filter/AE'
    os.environ['PYENSEMBL_CACHE_DIR'] = '/scratch/sys.argv[6]'

    if not (path_to_original_data.is_dir()):
        print(f'{path_to_original_data} is not a directory')
        sys.exit(-1)

    if not (pathway_data.is_file()):
        print(f'{pathway_data} is not a file')
        sys.exit(-1)

    pathways = pd.read_csv(pathway_data)
    pathways['All_Genes'] = pathways['All_Genes'].map(lambda x:
                                                      get_gene_ids_from_string(ensembl_release=ensembl_version,
                                                                               genes=x))
    pathways.to_csv(pathway_data.parent.joinpath('pathways_geneids.csv'))
    for filename in path_to_original_data.cwd().glob('*.csv'):
        geno = pd.read_csv(filename, index_col=0)  # original data
        # filter data
        for index, gene_set in enumerate(pathways.All_Genes):
            pathway = pathways.iloc[index + 1, 0]
            input_data = geno[geno.columns.intersection(gene_set)]
            base_name = f'{pathway}-{filename.stem}'
            job_directory = Path(f'{os.getcwd()}/.job')
            filtered_data_dir = path_to_save_filtered_data.joinpath(base_name)
            output_dir = filename.parent.joinpath(base_name)

            # Make top level directories
            mkdir_p(job_directory)
            mkdir_p(filtered_data_dir)
            mkdir_p(output_dir)

            job_file = job_directory.joinpath(f'{base_name}.job')
            output_data = output_dir.joinpath(f'{base_name}.csv')
            path_to_save_filtered_data = filtered_data_dir.joinpath(f'{base_name}.csv')

            input_data.to_csv(path_to_save_filtered_data)

            # process filtered dataset
            with open(job_file) as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH --partition=gpu-v100\n")
                fh.writelines("#SBATCH --gres=gpu:1\n")
                fh.writelines("#SBATCH --time=2:0:0\n")
                fh.writelines("#SBATCH --mem=8GB\n")
                fh.writelines("#SBATCH --job-name=%x-job-%N-%j.slurm.job\n")
                fh.writelines("#SBATCH --out=%x-job-%N-%j.slurm.out\n")
                fh.writelines("#SBATCH --error=%x-job-%N-%j.slurm.error\n")
                fh.writelines("#SBATCH --mail-type=ALL\n")
                fh.writelines("#SBATCH --mail-user=$USER@ucalgary.ca\n")

                fh.writelines("\n####### Set environment variables ###############\n\n")
                fh.writelines("module load python/anaconda3-2018.12\n")
                fh.writelines("conda activate XAI\n")

                fh.writelines("\n####### Run script ##############################\n\n")
                fh.writelines("echo \"python " + "${pwd} src\\AutoEncoder.py " + f'{base_name}_AE_Geno ' +
                                                                                 f'{path_to_save_filtered_data} ' +
                                                                                 f'{path_to_save_filtered_data.stem}' +
                                                                                 f' _QC.csv {save_dir}\n')

            output = subprocess.run([sys.executable, '-c', f'sbatch {job_file}'],
                                    capture_output=True, text=True, check=True)

            print('####################')
            print('Return code:', output.returncode)
            print('Output:', output.stdout)
