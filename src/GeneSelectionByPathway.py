# Filters Dataset by KEGG Pathway and uses PyEnsembl to get EnsemblID
import concurrent.futures
import distutils

from pyensembl import EnsemblRelease
from pathlib import Path
from CommonTools import create_dir, get_filtered_data
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
            ids.append((gene_data.gene_ids_of_gene_name(gene_name=gene)[0]).replace('\'', ''))
        except ValueError:
            ids.append(gene)
    return np.array(ids)


def get_gene_names(ensembl_release: int, gene_list: np.ndarray) -> np.ndarray:
    gene_data = EnsemblRelease(release=ensembl_release, species='human', server='ftp://ftp.ensembl.org/')
    names = []
    for gene in gene_list:
        try:
            names.append((gene_data.gene_name_of_gene_id(gene)).replace('\'', ''))
        except ValueError:
            names.append(gene)
    return np.array(names)


def create_sbatch_files(job_file, base_name, path_to_save_filtered_data, qc_file_gene_id, save_dir, base_bar_path,
                        qc_file_gene_name, base_scatter_path, base_model_path):
    # process filtered dataset
    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --mail-user=%uR@ucalgary.ca\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --partition=gpu-v100\n")
        fh.writelines("#SBATCH --gres=gpu:1\n")
        fh.writelines("#SBATCH --time=2:0:0\n")
        fh.writelines("#SBATCH --mem=16GB\n")
        fh.writelines("#SBATCH --cpus-per-task=8")
        fh.writelines(f"#SBATCH --job-name={base_name}-job.slurm\n")
        fh.writelines("#SBATCH --out=%x-%N-%j.out\n")
        fh.writelines("#SBATCH --error=%x-%N-%j.error\n")

        fh.writelines("\n####### Set environment variables ###############\n\n")
        fh.writelines("module load python/anaconda3-2019.10-tensorflowgpu\n")
        fh.writelines("source $HOME/.bash_profile\n")
        fh.writelines("conda activate XAI\n")

        fh.writelines("\n####### Run script ##############################\n")
        fh.writelines(f"echo \"python src/AutoEncoder.py {base_name}_AE_Geno " +
                      f"{path_to_save_filtered_data} {qc_file_gene_id} {save_dir} 64\"\n")
        fh.writelines(f"python src/AutoEncoder.py {base_name}_AE_Geno {path_to_save_filtered_data} " +
                      f"{qc_file_gene_id} {save_dir} 64\n")
        fh.writelines(f"echo \"python src/SHAP_combo.py {qc_file_gene_name} {qc_file_gene_id} {save_dir} "
                      f"{base_bar_path} {base_scatter_path} {base_model_path}\"\n")
        fh.writelines(f"python src/SHAP_combo.py {qc_file_gene_name} {qc_file_gene_id} {save_dir} "
                      f"{base_bar_path} {base_scatter_path} {base_model_path}\n")

        fh.writelines("\n####### Clean up ################################\n")
        fh.writelines("module unload python/anaconda3-2019.10-tensorflowgpu\n")
        fh.close()
    print(f"sbatch {job_file}")
    output = subprocess.run(('sbatch', job_file), capture_output=True, text=True, check=True)

    print('####################')
    print('Return code:', output.returncode)
    print('Output:\n', output.stdout)
    print('Error:\n', output.stderr)


def create_model(base_name, path_to_save_filtered_data, qc_file_gene_id, save_dir, base_bar_path,
                 qc_file_gene_name, base_scatter_path, base_model_path):
    print("\n####### Run script ##############################\n")
    print(f"python src/AutoEncoder.py {base_name}_AE_Geno {path_to_save_filtered_data} " +
          f"{qc_file_gene_id} {save_dir} 64\n")
    print(f"python src/SHAP_combo.py {qc_file_gene_name} {qc_file_gene_id} {save_dir} "
          f"{base_bar_path} {base_scatter_path} {base_model_path}\n")
'''
    out = subprocess.run(('python', 'src/AutoEncoder.py',  f'{base_name}_AE_Geno', path_to_save_filtered_data,
                          qc_file_gene_id, save_dir, 32, 10000, 4096), capture_output=True, text=True, check=True)

    print('####################')
    print('Return code:', out.returncode)
    print('Output:\n', out.stdout)
    print('Error:\n', out.stderr)

    out = subprocess.run(('python', 'src/SHAP_combo.py', qc_file_gene_name, qc_file_gene_id, save_dir,
                          base_bar_path, base_scatter_path, base_model_path), capture_output=True, text=True, check=True)

    print('####################')
    print('Return code:', out.returncode)
    print('Output:\n', out.stdout)
    print('Error:\n', out.stderr)
'''


def process_pathways(slurm: bool, ensembl_version: int, filename: Path, pathways: pd.DataFrame,
                     base_to_save_filtered_data: Path, dir_to_model: Path):
    geno: pd.DataFrame = pd.read_csv(filename, index_col=0)  # original data
    for index, gene_set in enumerate(pathways.All_Genes):
        pathway = pathways.iloc[index, 0]
        input_data: pd.DataFrame = geno[geno.columns.intersection(gene_set)]
        base_name = f'{pathway}-{filename.stem}'
        filtered_data_dir = base_to_save_filtered_data.joinpath(filename.stem)
        save_dir = dir_to_model.joinpath(base_name)

        # Make top level directories
        create_dir(filtered_data_dir)
        create_dir(save_dir)

        path_to_save_filtered_data = filtered_data_dir.joinpath(f'{base_name}.csv')
        input_data.to_csv(path_to_save_filtered_data)

        qc_file_gene_id = filtered_data_dir.joinpath(f'{base_name}_gene_id_QC.csv')

        qc_file_gene_name = filtered_data_dir.joinpath(f'{base_name}_gene_name_QC.csv')
        names = get_gene_names(ensembl_release=ensembl_version, gene_list=np.array(input_data.columns))
        input_data.rename(dict(zip(np.array(input_data.columns), names)), axis='columns', inplace=True)
        get_filtered_data(input_data, qc_file_gene_name)
        base_bar_path: Path = save_dir.joinpath('shap/bar')
        base_scatter_path: Path = save_dir.joinpath('shap/scatter')
        base_model_path: Path = save_dir.joinpath('shap/model')
        if slurm:
            job_directory = Path(f'{os.getcwd()}/.job')
            create_dir(job_directory)
            job_file = job_directory.joinpath(f'{base_name}.job')
            create_sbatch_files(job_file, base_name, path_to_save_filtered_data, qc_file_gene_id, save_dir,
                                base_bar_path, qc_file_gene_name, base_scatter_path, base_model_path)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                executor.submit(create_model, (base_name, path_to_save_filtered_data, qc_file_gene_id, save_dir,
                                               base_bar_path, qc_file_gene_name, base_scatter_path, base_model_path))


def get_pathways_gene_names(ensembl_version: int, pathway_data: Path) -> pd.DataFrame:
    pathways = pd.read_csv(pathway_data)
    pathways['All_Genes'] = pathways['All_Genes'].map(lambda x:
                                                      get_gene_ids_from_string(ensembl_release=ensembl_version,
                                                                               genes=x))
    pathways.to_csv(pathway_data.parent.joinpath('pathways_gene_ids.csv'))
    return pathways


def main(slurm: bool, ensembl_version: int, path_to_original_data: Path, pathway_data: Path,
         base_to_save_filtered_data: Path, dir_to_model: Path):
    if not (pathway_data.is_file()):
        print(f'{pathway_data} is not a file')
        sys.exit(-1)

    pathways = get_pathways_gene_names(ensembl_version=ensembl_version, pathway_data=pathway_data)
    for filename in path_to_original_data.glob('*.csv'):
        # filter data and create sbatch script
        process_pathways(slurm=slurm, ensembl_version=ensembl_version, filename=filename, pathways=pathways,
                         base_to_save_filtered_data=base_to_save_filtered_data, dir_to_model=dir_to_model)


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('less than 7 command line arguments')
        print('python GeneSelectionPathway.py ensemble_version dir_original_data '
              'filename_pathway_data dir_filtered_data dir_AE_model session_id')
        print('\tensembl_version - Ensembl Release version e.g. 104')
        print('\tdir_original_data - path to original data e.g. ./data/input/ or ./data_example.csv')
        print('\tfilename_pathway_data - filename of pathway data e.g. ./data/pathway.csv')
        print('\tdir_filtered_data - base dir to saved filtered original data e.g. ./data/filter')
        print('\tdir_AE_model - base dir to saved AE models e.g. .data/filter/AE')
        print('\tslurm - where to run program on slurm')
        print('\tsession_id - slurm job id')
        sys.exit(-1)

    # data setup
    tmp = distutils.util.strtobool(sys.argv[6])
    print(f'slurm: {sys.argv[6]} {bool(tmp)}')
    if bool(tmp):
        os.environ['PYENSEMBL_CACHE_DIR'] = '/scratch/' + sys.argv[7]
    if Path(sys.argv[2]).is_file():
        process_pathways(slurm=bool(tmp), ensembl_version=int(sys.argv[1]), filename=Path(sys.argv[2]),
                         pathways=get_pathways_gene_names(ensembl_version=int(sys.argv[1]),
                                                          pathway_data=Path(sys.argv[3])),
                         base_to_save_filtered_data=Path(sys.argv[4]), dir_to_model=Path(sys.argv[5]))
    else:
        main(bool(tmp), int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]),
             Path(sys.argv[5]))
