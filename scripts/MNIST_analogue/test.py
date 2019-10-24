import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import numpy as np
import torchvision
import sys
from functools import partial
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

quantize = Quantize.apply


class QuantizeLayer(nn.Module):
    def __init__(self, work=True):
        super().__init__()
        self.work = work

    def forward(self, data):
        if self.work:
            return quantize(data)
        else:
            return data


class QianClassifier(nn.Module):
    def __init__(self, qReLU=None):
        nn.Module.__init__(self)

        self.qReLU = qReLU

        self.conv1 = nn.Conv2d(1, 16, (5, 5), stride=1, padding=0, bias=False)
        self.pool1 = nn.AvgPool2d((2, 2))
        self.conv2 = nn.Conv2d(16, 64, (5, 5), stride=1, padding=0, bias=False)
        self.pool2 = nn.AvgPool2d((2, 2))
        self.dropout1 = nn.Dropout2d(0.5)
        self.dense1 = nn.Conv2d(64, 10, (4, 4), stride=1, padding=0, bias=False)
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            nn.init.kaiming_uniform_(w, nonlinearity='relu')
        print("Init weights with kaiming")

    def quantizeReLU(self, x):
        if self.qReLU == "sigmoid":
            x = torch.sigmoid(x - 0.5 - torch.floor(x)) + torch.floor(x)
        elif self.qReLU == "floor":
            x = Quantize.apply(x)
        return x

    def forward(self, data):
        x = nn.functional.relu(self.conv1(data))
        x = self.quantizeReLU(x)
        x = self.pool1(x) * self.pool1.kernel_size[0] * self.pool1.kernel_size[1]
        x = nn.functional.relu(self.conv2(x))
        x = self.quantizeReLU(x)
        x = self.pool2(x) * self.pool2.kernel_size[0] * self.pool2.kernel_size[1]
        x = self.dropout1(x)
        x = nn.functional.relu(self.dense1(x))
        x = self.quantizeReLU(x)
        y = x.view(-1, x.shape[-3] * x.shape[-2] * x.shape[-1])
        return y


def get_MAC_except_input_layer(model, i):
    MAC = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if ("conv" in name) or ("dense" in name):
                s = module.stride
                p = module.padding
                k = module.kernel_size
                o = module(i)
                MAC.append(i.shape[1] *
                           (i.shape[2] - k[0] + p[0] + 1) // s[0] *
                           (i.shape[3] - k[1] + p[1] + 1) // s[1] *
                           k[0] * k[1] *
                           o.shape[1])
                i = o
            elif "pool" in name:
                k = module.kernel_size
                o = module(i)
                i = o
        return np.sum(MAC[1:])


def get_activation(name, m, i, o):
    act = torch.floor(torch.nn.functional.relu(o))
    activation[name] = act


def get_synops(name, m, i, o):
    syn = i[0].sum(dim=tuple(range(1, i[0].ndim))) * m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    synops[name] = syn


def read_dict(dict_x, mode="sum"):
    scalar_x = dict()
    if mode == "sum":
        scalar_x["total"] = torch.tensor([0.]).cuda()
    for name, module in model.named_modules():
        if ("conv" in name) or ("dense" in name):
            if mode == "sum":
                x = dict_x[name]
                if x.ndim > 1:
                    x = x.sum(dim=tuple(range(1, x.ndim)))
                scalar_x[name] = x.mean()
                scalar_x["total"] += scalar_x[name]
            elif mode == "max":
                x = dict_x[name]
                if x.ndim > 1:
                    x = x.view(x.shape[0], -1)
                    x = x.max(dim=1)[0]
                scalar_x[name] = x.mean()
    return scalar_x

def register_model_hook(mdl, hook):
    for name, module in mdl.named_modules():
        if ("conv" in name) or ("dense" in name):
            module.register_forward_hook(partial(hook, name))
            
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

if __name__ == '__main__':
    import os
    import argparse
    from sinabs.from_torch import from_model

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--n_times', type=int, default=1)
    parser.add_argument('--quantize_method', type=str, default="floor")
    opt = parser.parse_args()

    quantize_method = opt.quantize_method
    model = QianClassifier(qReLU=quantize_method).cuda()

    input_tensor = torch.randn(1, 1, 28, 28).cuda()
    MACs = get_MAC_except_input_layer(model, input_tensor)
    print("%d MACs"%MACs)

    register_model_hook(model, get_activation)
    register_model_hook(model, get_synops)

    n_epochs = opt.n_epochs
    scale_down_synops = 0.5

    writer = SummaryWriter(log_dir=f"./runs/multiple")
    criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    BATCH_SIZE = 512

    # Load data
    mnist_dataset = MNIST('./data/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(mnist_dataset, batch_size=BATCH_SIZE, shuffle=True)

    step = 0
    for i_time in range(opt.n_times):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3/(1+i_time))
        pbar = tqdm(range(n_epochs))
        activation = dict()
        synops = dict()
        for epoch in pbar:
            for data, label in dataloader:
                step += 1
                optimizer.zero_grad()
                out = model(data.cuda())

                scalar_activation = read_dict(activation)
                scalar_synops = read_dict(synops)
                scalar_max_activation = read_dict(activation, "max")

                loss = criterion(out, label.cuda())
                scalar_synops["total"] -= scalar_synops["conv1"]
                synops_loss = scalar_synops["total"]
                preferred_synops = scale_down_synops * MACs
                if i_time > 0:
                    loss = ((synops_loss - preferred_synops) / preferred_synops) ** 2 + loss

                # # limit on max of activations
                # for key in scalar_max_activation:
                #     max_act = scalar_max_activation[key]
                #     if "dense" in key:
                #         preferred_mean_act = 1
                #     else:
                #         preferred_mean_act = 4
                #     loss += ((max_act - preferred_mean_act) / max_act)**2

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item(), activation=scalar_activation["total"].item(),
                                 synops=scalar_synops["total"].item())
                writer.add_scalar("loss", loss.item(), step)
                writer.add_scalars("activation", scalar_activation, step)
                writer.add_scalars("max_activation", scalar_max_activation, step)
                writer.add_scalars("synops", scalar_synops, step)



        dirName = "models"
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory ", dirName, " Created ")
        else:
            print("Directory ", dirName, " already exists")

        savefile = f'models/multiple_{i_time}.pth'
        torch.save(model.state_dict(), savefile)
        print(f"Model saved at {savefile}")

        # Test model
        test_dataset= MNIST('./data/', train=False, download=True, transform=torchvision.transforms.ToTensor())
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

        # Test model
        state_dict = model.state_dict()
        test_model = QianClassifier(qReLU="floor").cuda()
        register_model_hook(test_model, get_synops)
        test_model.load_state_dict(state_dict)

        all_pred = []
        all_synops = []
        with torch.no_grad():
            test_model.eval()
            for data, label in test_dataloader:
                out = test_model(data.cuda())
                _, pred = out.max(1)
                all_pred.append((pred == label.cuda()).float().mean().item())
                scalar_synops = read_dict(synops)
                scalar_synops["total"] -= scalar_synops["conv1"]
                all_synops.append(scalar_synops["total"].mean().item())
        ann_accuracy = np.mean(all_pred)
        ann_synops = np.mean(all_synops)
        print("Test accuracy after quantize: ", ann_accuracy)
        print("Test Sops after quantize: ", ann_synops)
        MACs = ann_synops

        # Test with SINABS on spiking simulation
        torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = from_model(
            test_model,
            (1, 28, 28),
            threshold=1.,
            membrane_subtract=1.,
            threshold_low=-1.,
            exclude_negative_spikes=True,
        ).to(torch_device)
        net.spiking_model.eval()

        all_pred = []
        all_synops = []
        spiking_test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        n_dt = 10

        with torch.no_grad():
            # loop over the input files
            for i, sample in enumerate(tqdm(spiking_test_dataloader)):
                # if i > 100: break
                test_data, test_labels = sample
                input_frames = tile(test_data / n_dt, 0, n_dt).to(torch_device)
                # we reset the network when changing file
                net.reset_states()
                outputs = net.spiking_model(input_frames)
                synops_df = net.get_synops(0)
                all_synops.append(synops_df['SynOps'].sum())

                _, predicted = outputs.sum(0).max(0)
                correctness = (predicted == test_labels.to(torch_device))
                all_pred.append(correctness.cpu().numpy())
        snn_accuracy = np.mean(all_pred)
        snn_synops = np.mean(all_synops)
        print("Test accuracy after quantize: ", snn_accuracy)
        print("Test Sops after quantize: ", snn_synops)

        with open("training_log_tmp.txt", "a") as f:
            f.write(f"{ann_accuracy} {ann_synops} {snn_accuracy} {snn_synops} {scale_down_synops}\n")
    writer.close()