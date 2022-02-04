# Filters Dataset by KEGG Pathway and uses PyEnsembl to get EnsemblID
from pyensembl import EnsemblRelease
from pathlib import Path
from AutoEncoderModule import create_dir
import pandas as pd
import numpy as np
import subprocess
import sys
import os


def get_gene_ids_from_string(ensembl_release: int, genes: str) -> np.ndarray:
    return get_gene_ids(ensembl_release=ensembl_release, gene_list=np.array(genes.split(';')))


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


def main(ensembl_version: int, path_to_original_data: Path, pathway_data: Path, base_to_save_filtered_data: Path,
         dir_to_model: Path):
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
    pathways.to_csv(pathway_data.parent.joinpath('pathways_gene_ids.csv'))
    for filename in path_to_original_data.glob('*.csv'):
        geno = pd.read_csv(filename, index_col=0)  # original data
        # filter data
        for index, gene_set in enumerate(pathways.All_Genes):
            pathway = pathways.iloc[index, 0]
            input_data = geno[geno.columns.intersection(gene_set)]
            base_name = f'{pathway}-{filename.stem}'
            job_directory = Path(f'{os.getcwd()}/.job')
            filtered_data_dir = base_to_save_filtered_data.joinpath(base_name)
            save_dir = dir_to_model.joinpath(base_name)

            # Make top level directories
            create_dir(job_directory)
            create_dir(filtered_data_dir)
            create_dir(save_dir)

            job_file = job_directory.joinpath(f'{base_name}.job')
            path_to_save_filtered_data = filtered_data_dir.joinpath(f'{base_name}.csv')
            qc_file = filtered_data_dir.joinpath(f'{base_name}_QC.csv')

            input_data.to_csv(path_to_save_filtered_data)

            # process filtered dataset
            with open(job_file, "w") as fh:
                fh.writelines("#!/bin/bash\n")
                fh.writelines("#SBATCH --mail-user=%uR@ucalgary.ca\n")
                fh.writelines("#SBATCH --mail-type=ALL\n")
                fh.writelines("#SBATCH --partition=gpu-v100\n")
                fh.writelines("#SBATCH --gres=gpu:1\n")
                fh.writelines("#SBATCH --time=2:0:0\n")
                fh.writelines("#SBATCH --mem=8GB\n")
                fh.writelines(f"#SBATCH --job-name={base_name}-job.slurm\n")
                fh.writelines("#SBATCH --out=%x-%N-%j.out\n")
                fh.writelines("#SBATCH --error=%x-%N-%j.error\n")

                fh.writelines("\n####### Set environment variables ###############\n\n")
                fh.writelines("module load python/anaconda3-2019.10-tensorflowgpu\n")
                fh.writelines("source $HOME/.bash_profile\n")
                fh.writelines("conda activate XAI\n")

                fh.writelines("\n####### Run script ##############################\n")
                fh.writelines("echo \"python -m src AutoEncoder.py {base_name}_AE_Geno " +
                              f"{path_to_save_filtered_data} {qc_file} {save_dir} 64\"\n")
                fh.writelines(f"python -m src AutoEncoder.py {base_name}_AE_Geno {path_to_save_filtered_data} " +
                              f"{qc_file} {save_dir} 64\\n")

                fh.writelines("\n####### Clean up ################################\n")
                fh.writelines("module unload python/anaconda3-2019.10-tensorflowgpu\n")
                fh.close()

            output = subprocess.run(('sbatch', job_file), capture_output=True, text=True, check=True)

            print('####################')
            print('Return code:', output.returncode)
            print('Output:', output.stdout)


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('less than 6 command line arguments')
        print('python GeneSelectionPathway.py ensemble_version dir_original_data '
              'filename_pathway_data dir_filtered_data dir_AE_model session_id')
        print('\tensembl_version - Ensembl Release version e.g. 104')
        print('\tdir_original_data - path to original data e.g. ./data/input/')
        print('\tfilename_pathway_data - filename of pathway data e.g. ./data/pathway.csv')
        print('\tdir_filtered_data - base dir to saved filtered original data e.g. ./data/filter')
        print('\tdir_AE_model - base dir to saved AE models e.g. .data/filter/AE')
        print('\tsession_id - slurm job id')
        sys.exit(-1)

    # data setup
    os.environ['PYENSEMBL_CACHE_DIR'] = '/scratch/' + sys.argv[6]
    main(int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
