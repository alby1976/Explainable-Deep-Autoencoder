from typing import Any, Union
from pathlib import Path
from numpy import ndarray
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class GPDataSet(Dataset):
    def __init__(self, gp_list):
        # 'Initialization'
        self.gp_list = gp_list

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.gp_list)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Load data and get label
        x = self.gp_list[index]
        x = np.array(x)
        return x


class AutoGenoShallow(nn.Module):
    def __init__(self, input_features, hidden_layer, smallest_layer, output_features):
        super().__init__()  # I guess this inherits __init__ from super class

        # def the encoder function
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_layer),
            nn.ReLU(True),
            nn.Linear(hidden_layer, smallest_layer),
            nn.ReLU(True),
        )

        # def the decoder function
        self.decoder = nn.Sequential(
            nn.Linear(smallest_layer, hidden_layer),
            nn.ReLU(True),
            nn.Linear(hidden_layer, output_features),
            nn.Sigmoid()
        )

    def _forward_unimplemented(self, *inputs: Any) -> None:
        pass

    # def forward function
    def forward(self, x):
        y = self.encoder(x)
        x = self.decoder(y)
        return x, y


def create_dir(directory: Path):
    """make a directory (directory) if it doesn't exist"""
    directory.mkdir(parents=True, exist_ok=True)


def run_ae(model_name: str, model: AutoGenoShallow, geno_train_set_loader: DataLoader, geno_test_set_loader: DataLoader,
           input_features: int, optimizer: Adam, distance=nn.MSELoss(), num_epochs=200, batch_size=4096, do_train=True,
           do_test=True, save_dir: Path = Path('./model')):
    create_dir(Path(save_dir))
    for epoch in range(num_epochs):
        batch_precision_list = []
        output_coder_list = []
        average_precision = 0.0
        sum_loss = 0.0
        if do_train:
            current_batch: int = 0
            model.train()
            for geno_data in geno_train_set_loader:
                current_batch += 1
                train_geno = Variable(geno_data).float().cuda()
                # =======forward========
                output, coder = model.forward(train_geno)
                loss = distance(output, train_geno)
                sum_loss += loss.item()
                # ======get coder======
                coder2 = coder.cpu().detach().numpy()
                output_coder_list.extend(coder2)
                # ======precision======
                output2 = output.cpu().detach().numpy()
                output3 = np.floor(output2 * 3) / 2  # make output3's value to 0, 0.5, 1
                diff = geno_data.numpy() - output3  # [0,0.5,1] - [0.0, 0.5, 0.5]
                diff_num = np.count_nonzero(diff)
                batch_average_precision = 1 - diff_num / (batch_size * input_features)
                batch_precision_list.append(batch_average_precision)
                # ======backward========
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===========log============
            coder_np: Union[ndarray, int] = np.array(output_coder_list)
            coder_file = save_dir.joinpath(f"{model_name}-{str(epoch)}.csv")
            np.savetxt(fname=coder_file, X=coder_np, fmt='%f', delimiter=',')
            average_precision = sum(
                batch_precision_list) / current_batch  # precision_list = [ave_pre_batch1, ave_pre_batch2,...]
        # ===========test==========
        test_batch_precision_list = []
        test_average_precision = 0.0
        test_sum_loss = 0.0
        if do_test:
            test_current_batch: int = 0
            model.eval()
            for geno_test_data in geno_test_set_loader:
                test_current_batch += 1
                test_geno = Variable(geno_test_data).float().cuda()
                # =======forward========
                test_output, coder = model.forward(test_geno)
                loss = distance(test_output, test_geno)
                test_sum_loss += loss.item()
                # ======precision======
                test_output2 = test_output.cpu().detach().numpy()
                test_output3 = np.floor(test_output2 * 3) / 2  # make output3's value to 0, 0.5, 1
                diff = geno_test_data.numpy() - test_output3  # [0,0.5,1] - [0.0, 0.5, 0.5]
                diff_num = np.count_nonzero(diff)
                batch_average_precision = 1 - diff_num / (batch_size * input_features)  # a single value
                test_batch_precision_list.append(batch_average_precision)  # [ave_pre_batch1, ave_pre_batch2,...]
            test_average_precision = sum(
                test_batch_precision_list) / test_current_batch
        print(f"epoch[{epoch + 1:3d}/{num_epochs}, loss: {sum_loss:.4f}, precision: {average_precision:.4f}, "
              f" test lost: {test_sum_loss:.4f}, test precision: {test_average_precision:.4f}")
