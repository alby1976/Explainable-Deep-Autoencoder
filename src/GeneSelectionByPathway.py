# Filters Dataset by KEGG Pathway and uses PyEnsembl to get EnsemblID
from pyensembl import EnsemblRelease
import pandas as pd
import numpy as np


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
    path_to_data = '../data_example.csv'  # path to original data
    path_to_pathway_data = './data_example_QC.csv'  # path to pathway data
    path_to_save_filtered_data = ' ./'  # path to saved filtered original data
    ensembl_version = 104  # Ensembl Release version

    geno = pd.read_csv(path_to_data, index_col=0)  # original data
    pathways = pd.read_csv(path_to_pathway_data)
    pathways.All_Genes = pathways.All_Genes.apply(lambda x:
                                                  get_gene_ids_from_string(ensembl_release=ensembl_version, genes=x))
    table = {}
    # filter data
    for index, gene_set in enumerate(pathways.All_Genes):
        table[pathways.iloc[index + 1, 0]] = geno[geno.columns.intersection(gene_set)]

    # process filtered dataset
