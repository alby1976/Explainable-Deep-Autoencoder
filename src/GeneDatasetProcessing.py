# Filters Dataset by KEGG Pathway and uses PyEnsembl to get EnsemblID
import argparse
import concurrent.futures
import distutils.util
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from CommonTools import create_dir, get_gene_ids, get_gene_names


def get_gene_ids_from_string(ensembl_release: int, genes: str) -> np.ndarray:
    return get_gene_ids(ensembl_release=ensembl_release, gene_list=np.array(genes.split(';')))


def create_sbatch_files(job_file, base_name, path_to_save_filtered_data, qc_file_gene_id, save_dir, base_bar_path,
                        qc_file_gene_name, base_scatter_path, base_model_path):
    # process filtered dataset
    line: list = []
    with open(job_file, "w") as fh:
        line.append("#!/bin/bash\n")
        line.append("#SBATCH --mail-user=%uR@ucalgary.ca\n")
        line.append("#SBATCH --mail-type=ALL\n")
        line.append("#SBATCH --partition=gpu-v100\n")
        line.append("#SBATCH --gres=gpu:1\n")
        line.append("#SBATCH --time=2:0:0\n")
        line.append("#SBATCH --mem=16GB\n")
        line.append("#SBATCH --cpus-per-task=8")
        line.append(f"#SBATCH --job-name={base_name}-job.slurm\n")
        line.append("#SBATCH --out=%x-%N-%j.out\n")
        line.append("#SBATCH --error=%x-%N-%j.error\n")

        line.append("\n####### Set environment variables ###############\n\n")
        line.append("module load python/anaconda3-2019.10-tensorflowgpu\n")
        line.append("source $HOME/.bash_profile\n")
        line.append("conda activate XAI\n")

        line.append("\n####### Run script ##############################\n")
        line.append(f"echo \"python src/AutoEncoder.py {base_name}_AE_Geno " +
                    f"{path_to_save_filtered_data} {qc_file_gene_id} {save_dir} 64\"\n")
        line.append(f"python src/AutoEncoder.py {base_name}_AE_Geno {path_to_save_filtered_data} " +
                    f"{qc_file_gene_id} {save_dir} 64\n")
        line.append(f"echo \"python src/SHAP_combo.py {qc_file_gene_name} {qc_file_gene_id} {save_dir} "
                    f"{base_bar_path} {base_scatter_path} {base_model_path}\"\n")
        line.append(f"python src/SHAP_combo.py {qc_file_gene_name} {qc_file_gene_id} {save_dir} "
                    f"{base_bar_path} {base_scatter_path} {base_model_path}\n")

        line.append("\n####### Clean up ################################\n")
        line.append("module unload python/anaconda3-2019.10-tensorflowgpu\n")
        fh.writelines(line)
        fh.flush()
        os.fsync(fd=fh)
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
          f"{qc_file_gene_id} {save_dir} 32 200 4096\n")
    out = subprocess.run(('python', 'src/AutoEncoder.py', f'{base_name}_AE_Geno', path_to_save_filtered_data,
                          qc_file_gene_id, save_dir, 32, 200, 4096), capture_output=True, text=True, check=True)

    print('####################')
    print('Return code:', out.returncode)
    print('Output:\n', out.stdout)
    print('Error:\n', out.stderr)

    print(f"python src/SHAP_combo.py {qc_file_gene_name} {qc_file_gene_id} {save_dir} "
          f"{base_bar_path} {base_scatter_path} {base_model_path}\n")
    out = subprocess.run(('python', 'src/SHAP_combo.py', qc_file_gene_name, qc_file_gene_id, save_dir,
                          base_bar_path, base_scatter_path, base_model_path),
                         capture_output=True, text=True, check=True)

    print('####################')
    print('Return code:', out.returncode)
    print('Output:\n', out.stdout)
    print('Error:\n', out.stderr)


def merge_gene(slurm: bool, ensembl_version: int, geno: pd.DataFrame, filename: Path, pathways: pd.DataFrame,
               base_to_save_filtered_data: Path, dir_to_model: Path, index: int, gene_set):
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
            executor.submit(create_model, base_name, path_to_save_filtered_data, qc_file_gene_id, save_dir,
                            base_bar_path, qc_file_gene_name, base_scatter_path, base_model_path)


def process_pathways(slurm: bool, ensembl_version: int, filename: Path, pathways: pd.DataFrame,
                     base_to_save_filtered_data: Path, dir_to_model: Path):
    geno: pd.DataFrame = pd.read_csv(filename, index_col=0)  # original x
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(lambda x, y: merge_gene(slurm, ensembl_version, geno, filename, pathways,
                                             base_to_save_filtered_data, dir_to_model, index=x, gene_set=y),
                     enumerate(pathways.All_Genes))


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
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(lambda x: process_pathways(slurm=slurm, ensembl_version=ensembl_version, filename=x,
                                                pathways=pathways,
                                                base_to_save_filtered_data=base_to_save_filtered_data,
                                                dir_to_model=dir_to_model), path_to_original_data.glob('*.csv'))


if __name__ == '__main__':
    parser: ArgumentParser = argparse.ArgumentParser(description='Combining genes found in the cancer dataset with '
                                                                 'genes that exists in known pathway as input to the '
                                                                 'AE model.')
    parser.add_argument("--ensembl_version", type=int, default=104, help='Ensembl Release version e.g. 104')
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).absolute().parent.parent.joinpath("data_example.csv"),
                        help='original datafile e.g. ./data_example.csv')
    parser.add_argument("-td", "--transformed_data", type=Path,
                        default=Path(__file__).absolute().parent.parent.joinpath("data_QC.csv"),
                        help='filename of original x after quality control e.g. ./data_QC.csv')
    parser.add_argument("-sd", "--save_dir", type=Path,
                        default=Path(__file__).absolute().parent.parent.joinpath("AE"),
                        help='base dir to saved AE models e.g. ./AE')
    parser.add_argument("-pd", "--pathway_dir", type=Path, required=True,
                        help='pathway filename or directory e.g. data/pathway or data/pathway.csv')
    parser.add_argument("-fd", "--filter_dir", type=Path, required=True,
                        help='base dir to save cancer dataset and corresponding genes that exists in know pathway '
                             'e.g. ./x/filter')
    # TODO flesh out group and subparser
    """
    >> > subparsers = parser.add_subparsers(help='sub-command help')
    >> >
    >> >  # create the parser for the "a" command
    >> > parser_a = subparsers.add_parser('a', help='a help')
    >> > parser_a.add_argument('bar', type=int, help='bar help')
    >> >
    >> >  # create the parser for the "b" command
    >> > parser_b = subparsers.add_parser('b', help='b help')
    >> > parser_b.add_argument('--baz', choices='XYZ', help='baz help')
    >> >
    >> >  # parse some argument lists
    >> > parser.parse_args(['a', '12'])
    Namespace(bar=12, foo=False)
    >> > parser.parse_args(['--foo', 'b', '--baz', 'Z'])
    Namespace(baz='Z', foo=True)

    >> > parser = argparse.ArgumentParser(prog='PROG', add_help=False)
    >> > group = parser.add_argument_group('group')
    >> > group.add_argument('--foo', help='foo help')
    >> > group.add_argument('bar', help='bar help')
    >> > parser.print_help()
    usage: PROG[--foo FOO] bar

    group:
        bar    bar  help
        --foo  FOO  foo  help
    """

    subparser = parser.add_subparsers()
    slurm = subparser.add_parser(name="slurm")
    slurm.add_argument("id", type=int, required=True, help="slurm job id")
    slurm.add_argument('--slurm', action='store_true' default=False, help="whethere the system supoorts ")
    parser.add_subparsers()
    print('less than 7 command line arguments')
    print('python GeneSelectionPathway.py ensemble_version dir_original_data '
          'filename_pathway_data dir_filtered_data dir_AE_model session_id')
    print('\tensembl_version - Ensembl Release version e.g. 104')
    print('\tdir_original_data - path to original x e.g. ./x/input/ or ./data_example.csv')
    print('\tfilename_pathway_data - filename of pathway x e.g. ./x/pathway.csv')
    print('\tdir_filtered_data - base dir to saved filtered original x e.g. ./x/filter')
    print('\tdir_AE_model - base dir to saved AE models e.g. .x/filter/AE')
    print('\tslurm - where to run program on slurm')
    print('\tsession_id - slurm job id')
    parser.add_argument("-n", "--name", type=str, default='AE_Geno',
                        help='model name e.g. AE_Geno')
    parser.add_argument("-sd", "--save_dir", type=Path,
                        default=Path(__file__).absolute().parent.parent.joinpath("AE"),
                        help='base dir to saved AE models e.g. ./AE')
    parser.add_argument("-cr", "--ratio", type=int, default=8,
                        help='compression ratio for smallest layer NB: ideally a number that is power of 2')
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
                        help='the base learning rate for training e.g 0.0001')
    parser.add_argument("--fold", type=bool, default=False,
                        help='selecting this flag causes the x to be transformed to change fold relative to '
                             'row median. default is False')

    if len(sys.argv) < 6:
        print('less than 7 command line arguments')
        print('python GeneSelectionPathway.py ensemble_version dir_original_data '
              'filename_pathway_data dir_filtered_data dir_AE_model session_id')
        print('\tensembl_version - Ensembl Release version e.g. 104')
        print('\tdir_original_data - path to original x e.g. ./x/input/ or ./data_example.csv')
        print('\tfilename_pathway_data - filename of pathway x e.g. ./x/pathway.csv')
        print('\tdir_filtered_data - base dir to saved filtered original x e.g. ./x/filter')
        print('\tdir_AE_model - base dir to saved AE models e.g. .x/filter/AE')
        print('\tslurm - where to run program on slurm')
        print('\tsession_id - slurm job id')
        sys.exit(-1)

    # x setup
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
