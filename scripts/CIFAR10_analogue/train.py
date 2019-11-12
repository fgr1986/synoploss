from model import CIFAR10AnalogueClassifier
from tqdm import tqdm
from synoploss import SynOpLoss
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from sinabs.from_torch import from_model
# from torch.optim.lr_scheduler import StepLR, MultiStepLR

import torchvision.transforms as ttr
import numpy as np
import torch
import os
import argparse


# Define the function to count the correct prediction
def count_correct(output, target):
    _, predicted = torch.max(output, 1)
    acc = (predicted == target).sum().float()
    return acc.cpu().numpy()


def get_MAC_except_input_layer(model, i):
    # caculate the MAC operations for a torch model (excluding the first Conv)
    # model: pytorch model
    # i: input tensor

    MAC = []
    with torch.no_grad():
        for module in model.seq:
            if isinstance(module, torch.nn.Conv2d):
                s = module.stride
                p = module.padding
                k = module.kernel_size
                o = module(i)
                MAC.append(
                    i.shape[1]
                    * (i.shape[2] - k[0] + p[0] + 1)
                    // s[0]
                    * (i.shape[3] - k[1] + p[1] + 1)
                    // s[1]
                    * k[0]
                    * k[1]
                    * o.shape[1]
                )
                i = o
            elif isinstance(module, torch.nn.Linear):
                o = module(i)
                MAC.append(i.shape[1] * o.shape[1])
                i = o
            else:
                o = module(i)
                i = o
        return np.sum(MAC[1:])


# Define tensor_tile function to generate sequence of input images
def tensor_tile(a, dim, n_tile):
    # a: input tensor
    # dim: tile on a specific dim or dims in a tuple
    # n_tile: number of tile to repeat
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) +
                        i for i in range(init_dim)])
    )
    return torch.index_select(a, dim, order_index)


def prepare():
    # Setting up environment

    # Declare global environment parameters
    # Torch device: GPU or CPU
    # Torch dataloader: training
    # Torch dataloader: testing
    # Torch dataloader: spiking testing
    # Input image size: (n_channel, width, height)
    global device, \
        train_dataloader, \
        test_dataloader, \
        spiking_test_dataloader, \
        input_image_size

    # Torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model folder to save trained models
    os.makedirs("models", exist_ok=True)

    # Setting up random seed to reproduce experiments
    torch.manual_seed(0)
    if device is not "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    transform_train = ttr.Compose([
        ttr.RandomCrop(32, padding=4),
        ttr.RandomHorizontalFlip(),
        ttr.ToTensor(),
        ttr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = ttr.Compose([
        ttr.ToTensor(),
        ttr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Downloading/Loading MNIST dataset as tensors for training
    train_dataset = CIFAR10(
        "./data/",
        train=True,
        download=True,
        transform=transform_train,
    )

    # Downloading/Loading MNIST dataset as tensors for testing
    test_dataset = CIFAR10(
        "./data/",
        train=False,
        download=True,
        transform=transform_test,
    )

    # Define Torch dataloaders for training, testing and spiking testing
    BATCH_SIZE = 512
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    spiking_test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Define the size of input images
    input_image_size = (3, 32, 32)

    # Return global prameters
    return device, \
           train_dataloader, \
           test_dataloader, \
           spiking_test_dataloader, \
           input_image_size


def train(model, n_epochs=350, b_opt_syn=True, target_synops=0):
    # Training a CNN model
    print("Target synops: %d" % target_synops)

    # Define loss
    criterion = torch.nn.CrossEntropyLoss()
    synops_criterion = SynOpLoss(model.modules())
    # Define optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.25e-3, weight_decay=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25e-3, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[int(200. / 350. * n_epochs),
                                                   int(250. / 350. * n_epochs),
                                                   int(300. / 350. * n_epochs)], gamma=0.1)
    # Visualize and display training loss in a progress bar
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        # Count correct prediction and total test number
        n_correct = 0
        n_test = 0

        # over batches
        for imgs, labels in train_dataloader:
            # reset grad to zero for each batch
            optimizer.zero_grad()
            # port to device
            imgs, labels = imgs.to(device), labels.to(device)
            # forward pass
            outputs = model(imgs)
            # calculate loss
            target_loss = criterion(outputs, labels)
            synops_loss = synops_criterion()
            synops_relative_loss = ((synops_loss - target_synops) / target_synops) ** 2
            # calculate accuracy
            n_correct += count_correct(outputs, labels)
            n_test += len(labels)
            # add loss of synops optimization
            if b_opt_syn and (target_synops > 0):
                loss = target_loss + synops_relative_loss
                # loss = synops_loss # * 0.001
            else:
                loss = target_loss
            # display in pbar
            pbar.set_postfix(
                accuracy_traing=n_correct / n_test,
                loss=loss.item(),
                synops=synops_loss.item(),
                target_loss=target_loss.item(),
                synops_loss=synops_relative_loss.item(),
            )
            # backprop
            loss.backward()
            optimizer.step()
            # print(model.seq[1].weight.grad.mean())

        # write to tensorboard
        writer.add_scalar("accuracy_traing", n_correct / n_test, epoch)
        writer.add_scalar("loss", target_loss.item() + synops_relative_loss.item(), epoch)
        writer.add_scalar("synops", synops_loss.item(), epoch)
        writer.add_scalar("target_loss", target_loss.item(), epoch)
        writer.add_scalar("synops_loss", synops_loss.item(), epoch)

        # test on model
        test(model, b_quantize=model.quantize, b_last_layer_relu=model.last_layer_relu, b_validation=True)
        # learning rate scheduler
        scheduler.step()
    return model


def test(model, b_quantize=True, b_last_layer_relu=True, b_validation=False):
    test_model = CIFAR10AnalogueClassifier(quantize=b_quantize, last_layer_relu=b_last_layer_relu).to(device)
    state_dict = model.state_dict()
    test_model.load_state_dict(state_dict)

    # Define loss
    criterion = torch.nn.CrossEntropyLoss()
    synops_criterion = SynOpLoss(test_model.modules())

    # With no gradient means less memory and calculation on forward pass
    with torch.no_grad():
        # Count correct prediction and total test number
        n_correct = 0
        n_test = 0
        n_synops = []
        f_target_loss = []

        # over batches
        for imgs, labels in test_dataloader:
            # evaluation usese Dropout and BatchNorm in inference mode
            test_model.eval()
            # port to device
            imgs, labels = imgs.to(device), labels.to(device)
            # inference
            outputs = test_model(imgs)
            n_correct += count_correct(outputs, labels)
            n_test += len(labels)
            n_synops.append(synops_criterion().item())

            if b_validation:
                test_model.train()
                outputs = test_model(imgs)
                target_loss = criterion(outputs, labels)
                f_target_loss.append(target_loss.item())

    ann_accuracy = n_correct / n_test
    ann_synops = np.mean(n_synops)
    print("ANN test accuracy: ", ann_accuracy)
    print("ANN test Sops: ", ann_synops)
    if b_validation:
        print("ANN target loss validation: ", np.mean(f_target_loss))
    return ann_accuracy, ann_synops


def snn_test(model, n_dt=10, n_test=10000):
    # Testing the accuracy of SNN on sinabs
    # model: CNN model
    # n_dt: the time window of each simulation
    # n_test: number of test images in total

    # Transfer Pytorch trained CNN model to sinabs SNN model
    net = from_model(
        model,  # Pytorch trained model
        input_image_size,  # Input image size: (n_channel, width, height)
        threshold=1.0,  # Threshold of the membrane potential of a Spiking neuron
        membrane_subtract=1.0,  # Subtract membrane potential when the neuron fires a spike
        threshold_low=-1.0,  # The lower bound of the membrane potential
        exclude_negative_spikes=True,  # Do not spike nagative spikes
    ).to(device)

    # With no gradient means less memory and calculation on forward pass
    with torch.no_grad():
        # evaluation usese Dropout and BatchNorm in inference mode
        net.spiking_model.eval()
        # Count correct prediction and total test number
        n_correct = 0
        all_synops = []
        # loop over the input files once a time
        for i, (imgs, labels) in enumerate(tqdm(spiking_test_dataloader)):
            if i > n_test:
                break
            # tile image to a sequence of n_dt length as input to SNN
            input_frames = tensor_tile(imgs / n_dt, 0, n_dt).to(device)
            labels = labels.to(device)
            # Reset neural states of all the neurons in the network for each inference
            net.reset_states()
            # inference
            outputs = net.spiking_model(input_frames)
            n_correct += count_correct(outputs.sum(0, keepdim=True), labels)
            # count synoptic operations
            synops_df = net.get_synops(0)
            all_synops.append(synops_df["SynOps"].sum())
    # calculate accuracy
    snn_accuracy = n_correct / n_test * 100.
    snn_synops = np.mean(all_synops)
    print("SNN test accuracy: %.2f" % (snn_accuracy))
    print("SNN test Sops: ", snn_synops)
    return snn_accuracy, snn_synops


def save_model(str_file_name, model):
    torch.save(model.state_dict(), str_file_name)
    print(f"Model saved at {str_file_name}")


def save_to_file(
        str_log_file, ann_accuracy, ann_synops, snn_accuracy, snn_synops, i_time
):
    with open(str_log_file, "a") as f:
        f.write(f"{ann_accuracy} {ann_synops} {snn_accuracy} {snn_synops} {i_time}\n")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    prepare()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=350)
    parser.add_argument("--n_times", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--b_save_model", type=str2bool, default=True)
    parser.add_argument("--str_log_file", type=str, default="log.txt")
    parser.add_argument("--target_scale", type=float, default=0.5)
    opt = parser.parse_args()

    print(opt)
    n_test = opt.n_test
    n_epochs = opt.n_epochs
    n_times = opt.n_times
    b_save_model = opt.b_save_model
    str_log_file = opt.str_log_file
    target_scale = opt.target_scale
    dropout_rate = (0.2, 0.5)

    # Original baseline of ANN: No quantise ReLU no Optimization on Synops
    classifier = CIFAR10AnalogueClassifier(quantize=False, dropout_rate=dropout_rate, last_layer_relu=False).to(device)
    writer = SummaryWriter(log_dir=f"./runs/Nov11_dr{dropout_rate[0]}_{dropout_rate[1]}_time_{-1}")
    str_file_name = f"models/Nov11_d{dropout_rate[0]}_{dropout_rate[1]}_{-1}.pth"
    # classifier = train(classifier, n_epochs=n_epochs, b_opt_syn=False)
    classifier.load_state_dict(torch.load(str_file_name))
    ann_accuracy, ann_synops = test(classifier, b_quantize=False, b_last_layer_relu=False)
    ann_accuracy, ann_synops = test(classifier, b_quantize=False)

    target_synops = get_MAC_except_input_layer(
        classifier, torch.randn(1, *(input_image_size)).to(device)
    )
    print(f"The MACs of this ANN model is {target_synops}")
    w_scale = target_synops / ann_synops
    print(f"weight scale = {w_scale}")
    if b_save_model:
        save_model(str_file_name, classifier)

    n_layers = len(list(classifier.parameters()))
    for i, w in enumerate(classifier.parameters()):
        if i < 1:
            w.data *= w_scale
    snn_accuracy, snn_synops = test(classifier, b_quantize=True)
    # snn_accuracy, snn_synops = snn_test(classifier, n_dt=10, n_test=n_test)
    save_to_file(str_log_file, ann_accuracy, ann_synops, snn_accuracy, snn_synops, -1)

    # Training with qReLU
    classifier = CIFAR10AnalogueClassifier(quantize=True, dropout_rate=(0.2, 0.1), last_layer_relu=True).to(device)
    classifier.load_state_dict(torch.load(str_file_name))
    n_layers = len(list(classifier.parameters()))
    ann_accuracy, ann_synops = test(classifier, b_quantize=False)
    for i, w in enumerate(classifier.parameters()):
        if i < 1:
            w.data *= w_scale
        elif i == n_layers - 1:
            w.data /= w_scale

    ann_accuracy, ann_synops = test(classifier, b_quantize=True)

    for i_time in range(n_times):
        if i_time == 0:
            n_retrain_epochs = n_epochs
        else:
            n_retrain_epochs = int(n_epochs / 10)
        writer = SummaryWriter(log_dir=f"./runs/Nov11_dr{dropout_rate[0]}_{dropout_rate[1]}_time_{i_time}")
        classifier = train(
            classifier,
            n_epochs=n_retrain_epochs,
            b_opt_syn=True,
            target_synops=target_synops,
        )
        ann_accuracy, ann_synops = test(classifier, b_quantize=True)
        snn_accuracy = ann_accuracy
        # snn_synops = ann_synops
        snn_accuracy, snn_synops = snn_test(classifier, n_dt=10, n_test=n_test)
        if (ann_synops / target_synops > 1.5) and (i_time > 0):
            target_scale /= 1.5
        target_synops = ann_synops * target_scale

        save_to_file(
            str_log_file, ann_accuracy, ann_synops, snn_accuracy, snn_synops, i_time
        )
        if b_save_model:
            str_file_name = f"models/Nov11_d{dropout_rate[0]}_{dropout_rate[1]}_{i_time}.pth"
            save_model(str_file_name, classifier)
