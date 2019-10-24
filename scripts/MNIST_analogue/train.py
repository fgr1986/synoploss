from model import MNISTAnalogueClassifier, BodoClassifier
from tqdm import tqdm
from synoploss import SynOpLoss
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from sinabs.from_torch import from_model

import numpy as np
import torchvision
import torch
import os
import argparse


def compute_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    acc = (predicted == target).sum().float() / len(target)
    return acc.cpu().numpy()


def get_MAC_except_input_layer(model, i):
    MAC = []
    with torch.no_grad():
        for module in model.seq:
            if isinstance(module, torch.nn.Conv2d):
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
            elif isinstance(module, torch.nn.Linear):
                o = module(i)
                MAC.append(i.shape[1] * o.shape[1])
                i = o
            else:
                o = module(i)
                i = o
        return np.sum(MAC[1:])


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def prepare():
    global BATCH_SIZE, device, \
           train_dataset, train_dataloader, \
           test_dataset, test_dataloader, \
           spiking_test_dataloader
    BATCH_SIZE = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('models', exist_ok=True)

    train_dataset = MNIST('./data/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset= MNIST('./data/', train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    spiking_test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


def train(model, n_epochs=20, f_lr_scale=1., b_opt_syn=True, n_last_synops=0):
    criterion = torch.nn.CrossEntropyLoss()
    synops_criterion = SynOpLoss(model.modules())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 * f_lr_scale)

    pbar = tqdm(range(n_epochs))
    preferred_synops = 0.5 * n_last_synops
    for epoch in pbar:
        model.train()
        accuracy_train = []
        for batch_id, sample in enumerate(train_dataloader):
            # if batch_id > 100: break
            optimizer.zero_grad()

            imgs, labels = sample
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            accuracy_train.append(compute_accuracy(outputs, labels))

            target_loss = criterion(outputs, labels)
            synops_loss = synops_criterion()
            loss = target_loss
            if b_opt_syn and (preferred_synops > 0):
                loss += ((synops_loss - preferred_synops) / preferred_synops) ** 2
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=target_loss.item(), synops=synops_loss.item())
        accuracy_train = np.mean(accuracy_train)
        pbar.set_postfix(loss=target_loss.item(), synops=synops_loss.item(), accuracy=accuracy_train)


def test(model, b_quantize=True):
    test_model = MNISTAnalogueClassifier(quantize=b_quantize).to(device)
    # test_model = BodoClassifier(quantize=b_quantize).to(device)
    state_dict = model.state_dict()
    test_model.load_state_dict(state_dict)
    all_pred = []
    all_synops = []
    synops_test_criterion = SynOpLoss(test_model.modules())
    with torch.no_grad():
        test_model.eval()
        for data, label in test_dataloader:
            out = test_model(data.to(device))
            all_pred.append(compute_accuracy(out, label.to(device)))
            synops_loss = synops_test_criterion()
            all_synops.append(synops_loss.item())
    ann_accuracy = np.mean(all_pred)
    ann_synops = np.mean(all_synops)
    print("ANN test accuracy: ", ann_accuracy)
    print("ANN test Sops: ", ann_synops)
    return ann_accuracy, ann_synops


def snn_test(model, n_dt=10, n_test=10000):
    net = from_model(
        model,
        (1, 28, 28),
        threshold=1.,
        membrane_subtract=1.,
        threshold_low=-1.,
        exclude_negative_spikes=True,
    ).to(device)
    net.spiking_model.eval()
    all_pred = []
    all_synops = []

    with torch.no_grad():
        # loop over the input files
        for i, sample in enumerate(tqdm(spiking_test_dataloader)):
            if i > n_test: break
            test_data, test_labels = sample
            input_frames = tile(test_data / n_dt, 0, n_dt).to(device)
            # we reset the network when changing file
            net.reset_states()
            outputs = net.spiking_model(input_frames)
            synops_df = net.get_synops(0)
            all_synops.append(synops_df['SynOps'].sum())

            _, predicted = outputs.sum(0).max(0)
            correctness = (predicted == test_labels.to(device))
            all_pred.append(correctness.cpu().numpy())
    snn_accuracy = np.mean(all_pred)
    snn_synops = np.mean(all_synops)
    print("SNN test accuracy: ", snn_accuracy)
    print("SNN test Sops: ", snn_synops)
    return snn_accuracy, snn_synops


def save_model(model, i_file):
    savefile = f'models/MNIST_{i_file}.pth'
    torch.save(model.state_dict(), savefile)
    print(f"Model saved at {savefile}")


def save_to_file(str_log_file, ann_accuracy, ann_synops, snn_accuracy, snn_synops, i_time):
    with open(str_log_file, "a") as f:
        f.write(f"{ann_accuracy} {ann_synops} {snn_accuracy} {snn_synops} {i_time}\n")


if __name__ == '__main__':
    prepare()

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--n_times', type=int, default=30)
    parser.add_argument('--n_test', type=int, default=10000)
    parser.add_argument('--b_save_model', type=bool, default=True)
    parser.add_argument('--str_log_file', type=str, default="log.txt")
    opt = parser.parse_args()

    n_test = opt.n_test
    n_epochs = opt.n_epochs
    n_times = opt.n_times
    b_save_model = opt.b_save_model
    str_log_file = opt.str_log_file

    # Original baseline of ANN: No quantise ReLU no Optimization on Synops
    classifier = MNISTAnalogueClassifier(quantize=False).to(device)
    # classifier = BodoClassifier(quantize=False).to(device)
    train(classifier, n_epochs=n_epochs, f_lr_scale=1., b_opt_syn=False)
    ann_accuracy, ann_synops = test(classifier, b_quantize=False)
    snn_accuracy, snn_synops = snn_test(classifier, n_dt=10, n_test=n_test)

    if b_save_model:
        save_model(classifier, -1)
    save_to_file(str_log_file, ann_accuracy, ann_synops, snn_accuracy, snn_synops, -1)
    # Training with qReLU
    classifier = MNISTAnalogueClassifier(quantize=True).to(device)
    # classifier = BodoClassifier(quantize=True).to(device)
    n_last_synops = get_MAC_except_input_layer(classifier, torch.randn(1, 1, 28, 28).to(device))
    print(f"The MACs of this ANN model is {n_last_synops}")

    for i_time in range(n_times):
        train(classifier, n_epochs=n_epochs, f_lr_scale=1./(i_time+1), b_opt_syn=i_time, n_last_synops=n_last_synops)
        ann_accuracy, ann_synops = test(classifier, b_quantize=True)
        snn_accuracy, snn_synops = snn_test(classifier, n_dt=10, n_test=n_test)
        n_last_synops = ann_synops
        save_to_file(str_log_file, ann_accuracy, ann_synops, snn_accuracy, snn_synops, i_time)
        if b_save_model:
            save_model(classifier, i_time)


