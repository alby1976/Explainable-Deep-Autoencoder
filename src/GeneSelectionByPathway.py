# Filters Dataset by KEGG Pathway and uses PyEnsembl to get EnsemblID
from numpy import ndarray
from pyensembl import EnsemblRelease
import pandas as pd
import numpy as np


def get_gene_list(genes: str) -> np.ndarray:
    return np.fromstring(genes, dtype=int, sep=' ')


# data setup
PATH_TO_DATA = '../data_example.csv'  # path to original data
PATH_TO_PATHWAY_DATA = './data_example_QC.csv'  # path to pathway data
PATH_TO_SAVE_FILTERED_DATA = ' '  # path to saved filtered original data

geno = pd.read_csv(PATH_TO_DATA, index_col=0)  # original data
pathway = pd.read_csv(PATH_TO_PATHWAY_DATA)
gene_list: ndarray = get_gene_list(pathway.All_Genes)
gene_data = EnsemblRelease(104)
ids = np.array([])
for gene in gene_list:
    np.append(ids, gene_data.gene_ids_of_gene_name(gene_name=gene)[0])
