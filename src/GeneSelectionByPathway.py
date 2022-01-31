# Filters Dataset by KEGG Pathway and uses PyEnsembl to get EnsemblID
import pathlib

from pyensembl import EnsemblRelease
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import sys
import os


def get_gene_ids_from_string(ensembl_release: int, genes: str) -> np.ndarray:
    return get_gene_ids(ensembl_release=ensembl_release, gene_list=np.fromstring(genes, dtype=str, sep=';'))


def get_gene_ids(ensembl_release: int, gene_list: np.ndarray) -> np.ndarray:
    gene_data = EnsemblRelease(ensembl_release)
    ids = []
    for gene in gene_list:
        ids.append(gene_data.gene_ids_of_gene_name(gene_name=gene)[0])
    return np.array(ids)


def mkdir_p(directory: pathlib.Path):
    """make a directory (directory) if it doesn't exist"""
    directory.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # data setup
    ensembl_version = int(sys.argv[1])  # Ensembl Release version e.g. 104
    path_to_original_data = Path(sys.argv[2])  # path to original data e.g. './data/input/'
    pathway_data = Path(sys.argv[3])  # pathway data e.g. './data/pathway.csv'
    path_to_save_filtered_data = Path(sys.argv[4])  # base dir to saved filtered original data e.g. './data/filter'
    save_dir = Path(sys.argv[5])  # base directory to save AE models

    if not (path_to_original_data.is_dir()):
        print(f'{path_to_original_data} is not a directory')
        sys.exit(-1)

    if not (pathway_data.is_file()):
        print(f'{pathway_data} is not a file')
        sys.exit(-1)

    pathways = pd.read_csv(pathway_data)
    for filename in path_to_original_data.cwd().glob('*.csv'):
        geno = pd.read_csv(filename, index_col=0)  # original data
        pathways.All_Genes = pathways.All_Genes.apply(lambda x:
                                                      get_gene_ids_from_string(ensembl_release=ensembl_version,
                                                                               genes=x))

        # filter data
        for index, gene_set in enumerate(pathways.All_Genes):
            pathway = pathways.iloc[index + 1, 0]
            input_data = geno[geno.columns.intersection(gene_set)]
            output_dir = filename.parent.joinpath(f'{pathway}-{filename.stem}')
            mkdir_p(output_dir)
            # process filtered dataset
            job_directory = "%s/.job" % os.getcwd()
            scratch = os.environ['HOME']
            data_dir = os.path.join(scratch, '/project/LizardLips')

            # Make top level directories
            mkdir_p(job_directory)
            mkdir_p(data_dir)

            lizards = ["LizardA", "LizardB"]

            for lizard in lizards:
                job_file = os.path.join(job_directory, "%s.job" % lizard)
                lizard_data = os.path.join(data_dir, lizard)

                # Create lizard directories
                mkdir_p(lizard_data)

                with open(job_file) as fh:
                    fh.writelines("#!/bin/bash\n")
                    fh.writelines("#SBATCH --job-name=%s.job\n" % lizard)
                    fh.writelines("#SBATCH --output=.out/%s.out\n" % lizard)
                    fh.writelines("#SBATCH --error=.out/%s.err\n" % lizard)
                    fh.writelines("#SBATCH --time=2-00:00\n")
                    fh.writelines("#SBATCH --mem=12000\n")
                    fh.writelines("#SBATCH --qos=normal\n")
                    fh.writelines("#SBATCH --mail-type=ALL\n")
                    fh.writelines("#SBATCH --mail-user=$USER@stanford.edu\n")
                    fh.writelines("Rscript $HOME/project/LizardLips/run.R %s potato shiabato\n" % lizard_data)

                command = ['python AutoEncoder.py', ensembl_version, filename, pathway_data,
                           path_to_save_filtered_data, save_dir]
                output = subprocess.run([sys.executable, '-c', f'sbatch {job_file}'],
                                        capture_output=True, text=True, check=True)
                print('####################')
                print('Return code:', output.returncode)
                # use decode function to convert to string
                print('Output:', output.stdout)
