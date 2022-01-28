# Filters Dataset by KEGG Pathway and uses PyEnsembl to get EnsemblID
from pyensembl import EnsemblRelease
import pandas as pd
import numpy as np


def get_gene_ids_from_string(genes: str) -> np.ndarray:
    return get_gene_ids(np.fromstring(genes, dtype=str, sep=';'))


def get_gene_ids(gene_list: np.ndarray) -> np.ndarray:
    gene_data = EnsemblRelease(104)
    ids = []
    for gene in gene_list:
        ids.append(gene_data.gene_ids_of_gene_name(gene_name=gene)[0])
    return np.array(ids)


# data setup
PATH_TO_DATA = '../data_example.csv'  # path to original data
PATH_TO_PATHWAY_DATA = './data_example_QC.csv'  # path to pathway data
PATH_TO_SAVE_FILTERED_DATA = ' '  # path to saved filtered original data

geno = pd.read_csv(PATH_TO_DATA, index_col=0)  # original data
pathways = pd.read_csv(PATH_TO_PATHWAY_DATA)
pathways.All_Genes = pathways.All_Genes.map(get_gene_ids)
table = {}
# filter data
for index, gene_set in enumerate(pathways.All_Genes):
    table[pathways.iloc[index + 1, 0]] = geno[geno.columns.intersection(gene_set)]

# process filtered dataset
