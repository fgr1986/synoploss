from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as ttr
from torch.utils.data import DataLoader

from model import MNISTClassifier
from aer4manager import AERFolderDataset


def compute_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    acc = (predicted == target).sum().float() / len(target)
    return acc.cpu().numpy()


# Parameters
BATCH_SIZE = 256

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Prepare datasets and dataloaders
train_dataset = AERFolderDataset(
    root='data/train/',
    from_spiketrain=False,
    transform=ttr.ToTensor(),
)

test_dataset = AERFolderDataset(
    root='data/test/',
    from_spiketrain=False,
    transform=ttr.ToTensor(),
)

print("Number of training frames:", len(train_dataset))
print("Number of testing frames:", len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

hookable_modules = ['seq.1', 'seq.4', 'seq.7', 'seq.12']
fanouts = {'seq.1': 72,
           'seq.4': 108,
           'seq.7': 432,
           'seq.12': 10,
           }


def train(penalty_coefficient, penalty_function,
          epochs=10, save=False, fanout_weighting=False,
          quantize_training=False):
    # Define model and learning parameters
    myclass = MNISTClassifier(quantize=quantize_training).to(device)
    # Define loss
    criterion = torch.nn.CrossEntropyLoss()
    # Define optimizer
    decay_rate = penalty_coefficient if WEIGHT_DECAY else 0.
    optimizer = torch.optim.Adam(myclass.parameters(), lr=1e-3,
                                 weight_decay=decay_rate)

    # Set hooks
    activation = torch.cuda.FloatTensor([0.])
    value_to_penalize = torch.cuda.FloatTensor([0.])

    def hook(m, i, o):
        nonlocal activation, value_to_penalize
        activation += o.sum() / BATCH_SIZE
        value_to_penalize += penalty_function(o) * m.fanout

    for name, module in myclass.named_modules():
        if name in hookable_modules:
            module.fanout = fanouts[name] if fanout_weighting else 1.
            module.register_forward_hook(hook)

    # Impose Kaiming He initialization
    for w in myclass.parameters():
        torch.nn.init.kaiming_uniform_(w, nonlinearity='relu')

    # Start training
    for epoch in range(epochs):
        myclass.train()
        accuracy_train = []

        print(f"Epoch {epoch}, training")
        for batch_id, sample in enumerate(tqdm(train_dataloader)):
            # if batch_id > 100: break
            optimizer.zero_grad()

            activation = torch.cuda.FloatTensor([0.])
            value_to_penalize = torch.cuda.FloatTensor([0.])

            imgs, labels = sample
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = myclass(imgs)
            accuracy_train.append(compute_accuracy(outputs, labels))

            target_loss = criterion(outputs, labels)
            loss = target_loss + value_to_penalize * penalty_coefficient
            loss.backward()
            optimizer.step()

        accuracy_train = np.mean(accuracy_train)
        print(f"loss={loss.item()}, accuracy_train={accuracy_train}")

    # Test network accuracy
    with torch.no_grad():
        myclass.eval()
        accuracy = []

        print(f"Epoch {epoch}, testing")
        for batch_id, sample in enumerate(tqdm(test_dataloader)):
            # if batch_id > 10: break
            test_data, test_labels = sample
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            outputs = myclass(test_data)
            accuracy.append(compute_accuracy(outputs, test_labels))

        accuracy = np.mean(accuracy)
        print(f"loss={loss.item()}, accuracy_test={accuracy}")

    # Save trained model
    if save:
        torch.save(myclass.state_dict(),
                   f'models/{save}_{penalty_coefficient}.pt')
        print(f"Model saved at {save}")

    return penalty_coefficient, activation.item(), accuracy, target_loss.item()


def launch_trainings(penalties, penalty_function, name, fanout=False,
                     quantize_training=False):
    res = []
    for p in penalties:
        res.append(train(p, penalty_function=penalty_function, epochs=N_EPOCHS,
                         save=name, fanout_weighting=fanout,
                         quantize_training=quantize_training))

    results = np.asarray(res)
    np.savetxt('results/' + name + '.txt', results)


def l2neuron_penalty(out):
    return (out.mean(0)**2).sum()


def l2layer_penalty(out):
    return (out.mean(0).sum())**2


def l1_penalty(out):
    return out.mean(0).sum()


def null_penalty(out):
    return 0.


if __name__ == '__main__':
    N_EPOCHS = 5
    WEIGHT_DECAY = False
    N_MODELS = 20

    # # L2 neuron-wise
    # penalties = np.logspace(-4, -0, N_MODELS)
    # name = "l2neuron"
    # launch_trainings(penalties, l2neuron_penalty, name)
    # # L2 layer-wise
    # penalties = np.logspace(-8, 0, N_MODELS)
    # name = "l2layer"
    # launch_trainings(penalties, l2layer_penalty, name)
    # # L1 penalty
    # penalties = np.logspace(-5, -2, N_MODELS)
    # name = "l1"
    # launch_trainings(penalties, l1_penalty, name)
    # # no penalty
    # penalties = [0.]
    # name = "nopenalty"
    # launch_trainings(penalties, null_penalty, name)
    # L1 penalty with fanout weighing
    penalties = np.logspace(-10, -4, N_MODELS)
    name = "l1fanout_qtrain"
    launch_trainings(penalties, l1_penalty, name, fanout=True,
                     quantize_training=True)
