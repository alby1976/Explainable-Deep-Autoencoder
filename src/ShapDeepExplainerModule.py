# python system library
# 3rd party modules
from concurrent.futures import ThreadPoolExecutor

from matplotlib import pyplot as plt

from AutoEncoderModule import AutoGenoShallow
from SHAP_combo import *
from CommonTools import *


def create_gene_model(model_name: str, gene_model: Path, shap_values, gene_names: ndarray, sample_num: int,
                      top_num: int, node: int):
    # **generate gene model
    shap_values_mean = np.sum(abs(shap_values), axis=0) / sample_num
    shap_values_ln = np.log(shap_values_mean)  # *calculate ln^|shap_values_mean|
    gene_module: Union[ndarray, DataFrame] = np.stack((gene_names, shap_values_ln), axis=0)
    gene_module = gene_module.T
    gene_module = gene_module[np.argsort(gene_module[:, 1])]
    gene_module = gene_module[::-1]  # [starting index: stopping index: stepcount]
    gene_module = pd.DataFrame(gene_module)
    gene_module = gene_module.head(top_num)
    masking: Union[ndarray, bool] = gene_module[[1]] != -np.inf
    gene_module = gene_module[masking.all(axis=1)]
    if len(gene_module.index) > (1 / 4) * top_num:
        filename = f"{model_name}-shap({node}).csv"
        print(f'Creating {gene_model.joinpath(filename)} ...')
        gene_module.to_csv(gene_model.joinpath(filename), header=True, index=False, sep='\t')
        tbl = wandb.Table(dataframe=gene_module)
        tmp = f"{model_name}-shap({node})"
        wandb.log({tmp: tbl})
        print("... Done ...\n")


def plot_shap_values(model_name: str, node: int, values, x_test: Union[ndarray, DataFrame, List], names: ndarray,
                     plot_type: str, plot_size, save_shap: Path):
    filename = f"{model_name}-{plot_type}({node}).png"
    print(f"Creating {save_shap.joinpath(filename)} ...")
    shap.summary_plot(values, x_test, names, plot_type=plot_type, plot_size=plot_size)
    plt.savefig(f"{save_shap.joinpath(filename)}", dpi=100, format='png')
    plt.close()
    tmp = f"{model_name}-{plot_type}({node})"
    wandb.log({tmp: wandb.Image(save_shap.joinpath(filename))})
    print("Done")


def process_shap_values(save_bar: Path, save_scatter: Path, gene_model: Path, model_name: str, x_test, shap_values,
                        gene_names, sample_num, top_num, node):
    print(f"save_bar: {save_bar}\nsave_scatter: {save_scatter}\ngene_model: {gene_model}\nmodel_name: {model_name}\n"
          f"x_test:\n{x_test}\nshap_values:\n{shap_values}\ngene_names:{gene_names}\n")
    # save shap_gene_model
    create_gene_model(model_name, gene_model, shap_values, gene_names, sample_num, top_num, node)

    # generate bar char
    plot_shap_values(model_name, node, shap_values, x_test, gene_names, "bar", (15, 10), save_bar)
    # generate scatter chart
    plot_shap_values(model_name, node, shap_values, x_test, gene_names, "dot", (15, 10), save_scatter)


def create_shap_values(model: AutoGenoShallow, model_name: str, gene_model: Path, save_bar: Path, save_scatter: Path,
                       top_rate: float = 0.05, num_workers: int = 8):
    shap_values: Union[
        ndarray, List[ndarray], Tuple[List[Union[ndarray, List[ndarray]]], Any], List[Union[ndarray, List[ndarray]]]]

    # setup
    create_dir(gene_model)
    create_dir(save_bar)
    create_dir(save_scatter)
    # model.decoder = nn.Identity()
    print(f"model type: {type(model)} device: {model.device}\n{model}\n\n")

    genes = torch.cat([batch[0] for batch in model.predict_dataloader()])
    x_train: Tensor = genes[:100]
    try:
        print(f"x_train type: {type(x_train)} device; cuda:{x_train.get_device()}")
    except RuntimeError:
        print(f"x_train type: {type(x_train)} device; cpu")
    print(f"{x_train}\n\n")

    x_test: Tensor = torch.cat([batch[0] for batch in model.val_dataloader()])
    try:
        print(f"x_train type: {type(x_test)} device; cuda:{x_test.get_device()}")
    except RuntimeError:
        print(f"x_train type: {type(x_test)} device; cpu")
    print(f"{x_test}\n\n")
    print(f"model type: {type(model)} device: {model.device}\n{model}\n\n")

    print(f"gene_names")
    gene_names: ndarray = model.dataset.gene_names[model.dataset.dm.column_mask]
    sample_size = x_test.size(dim=1)
    top_num: int = int(top_rate * len(gene_names))  # top_rate is the percentage of features to be calculated

    explainer = shap.DeepExplainer(model, x_train)
    shap_values = explainer.shap_values(x_test)  # shap_values contains values for all nodes
    # shap_values, top_index = explainer.shap_values(x_test, top_num, "max") # shap_values contains values for all nodes
    shap_values = np.asarray(shap_values)
    # top_index = np.swapaxes(top_index,0, 1)
    gene_table = wandb.Table(dataframe=pd.DataFrame(data=gene_names), columns=[i for i in range(sample_size)])
    wandb.log({"top num of features": top_num,
               "sample size": x_test.size(dim=0),
               "input features": sample_size,
               "gene name index": gene_table,
               "shap_values": shap_values.shape,
               # "top index": top_index.size()
               })

    # shap_table = {f"Shap Value Node{i}": wandb.Table(dataframe=pd.DataFrame(data=node, columns=gene_names))
    #              for i, node in enumerate(shap_values)}
    # top_table = {f"Top Shap Value rows{i}": wandb.Table(dataframe=pd.DataFrame(data=row.detach().cpu().numpy()))
    #             for i, row in enumerate(top_index)}
    # wandb.log(shap_table)
    # wandb.log(top_table)
    x_test = x_test.detach().cpu().numpy()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        params = ((save_bar, save_scatter, gene_model, model_name, x_test, shap_value, gene_names,
                   sample_size, top_num, node) for node, shap_value in enumerate(shap_values))
        print(f"params:\n{params}\n\n")
        # print(f"index:\n{top_index}\n\n")
        for _ in pool.map(lambda p: process_shap_values(*p), params):
            pass
        print("\n\t....Finish processing....")


def main(ckpt: Path, model_name: str, gene_model: Path, save_bar: Path, save_scatter: Path,
         top_rate: float = 0.05, num_workers: int = 8):
    # model = AutoGenoShallow()
    with wandb.init(name=model_name, project="XAE4Exp"):
        # wandb configuration
        wandb.config.update = {"architecture": platform.platform(),
                               "Note": f"Using DeepExplainer top {top_rate} of input features are explained"}

        model = AutoGenoShallow.load_from_checkpoint(str(ckpt))
        create_shap_values(model, model_name, gene_model, save_bar, save_scatter, top_rate, num_workers=num_workers)
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculates the shapey values for the AE model's output")
    add_shap_arguments(parser)
    parser.add_argument("-sd", "--save_dir", type=Path, required=True,
                        default=Path(__file__).absolute().parent.parent.joinpath("AE"),
                        help='base dir to saved Shap models e.g. ./AE/shap')
    parser.add_argument("--ckpt", type=str, required=True,
                        help='path to AutoEncoder checkpoint.  e.g. ckpt/model.ckpt')
    parser.add_argument("--name", type=str, required=True, help='AE model name')
    parser.add_argument("-w", "--num_workers", type=int, default=1,
                        help='number of processors used to run in parallel. -1 mean using all processor '
                             'available default is None')

    args = parser.parse_args()
    print(f"args:\n{args}")

    '''
    data = GPDataModule(data_dir=args.data, val_split=args.val_split, filter_str=args.filter_str)
    main(args.save_dir.joinpath(args.ckpt), args.name, data, args.save_dir.joinpath(args.gene_model),
         args.save_dir.joinpath(args.save_bar), args.save_dir.joinpath(args.save_scatter),
         args.top_rate, args.num_workers)
    '''
    main(args.save_dir.joinpath(args.ckpt), args.name, args.save_dir.joinpath(args.gene_model),
         args.save_dir.joinpath(args.save_bar), args.save_dir.joinpath(args.save_scatter),
         args.top_rate, args.num_workers)
