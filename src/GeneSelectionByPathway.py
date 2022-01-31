# Filters Dataset by KEGG Pathway and uses PyEnsembl to get EnsemblID
from pyensembl import EnsemblRelease
import pandas as pd
import numpy as np
import subprocess
import sys


def get_gene_ids_from_string(ensembl_release: int, genes: str) -> np.ndarray:
    return get_gene_ids(ensembl_release=ensembl_release, gene_list=np.fromstring(genes, dtype=str, sep=';'))


def get_gene_ids(ensembl_release: int, gene_list: np.ndarray) -> np.ndarray:
    gene_data = EnsemblRelease(ensembl_release)
    ids = []
    for gene in gene_list:
        ids.append(gene_data.gene_ids_of_gene_name(gene_name=gene)[0])
    return np.array(ids)


if __name__ == '__main__':
    # data setup
    ensembl_version = int(sys.argv[1])  # Ensembl Release version e.g. 104
    path_to_original_data = sys.argv[2]  # path to original data e.g. './data/input/'
    pathway_data = sys.argv[3]  # pathway data e.g. './data/pathway.csv'
    path_to_save_filtered_data = sys.argv[4]  # base directory to saved filtered original data e.g. './data/filter'
    save_dir = sys.argv[5]  # directory to save AE models

    geno = pd.read_csv(path_to_data, index_col=0)  # original data
    pathways = pd.read_csv(pathway_data)
    pathways.All_Genes = pathways.All_Genes.apply(lambda x:
                                                  get_gene_ids_from_string(ensembl_release=ensembl_version, genes=x))
    table = {}
    # filter data
    for index, gene_set in enumerate(pathways.All_Genes):
        table[pathways.iloc[index + 1, 0]] = geno[geno.columns.intersection(gene_set)]

    # process filtered dataset
    command = ['python AutoEncoder.py', ensembl_version, filename, path_to_pathway_data, path_to_save_filtered_data, save_dir]
    output = subprocess.run([sys.executable, '-c', f'sbatch {job_file}'], capture_output=True, text=True, check=True)
    print('####################')
    print('Return code:', output.returncode)
    # use decode function to convert to string
    print('Output:', output.stdout)

