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
BATCH_SIZE = 128

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
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def train(penalty_coefficient=0., epochs=10, save=False):
    # Define model and learning parameters
    myclass = MNISTClassifier().to(device)
    # Define loss
    criterion = torch.nn.CrossEntropyLoss()
    # Define optimizer
    decay_rate = penalty_coefficient if WEIGHT_DECAY else 0.
    optimizer = torch.optim.Adam(myclass.parameters(), lr=1e-3,
                                 weight_decay=decay_rate)

    # Set hooks
    hookable_modules = ['seq.1', 'seq.4', 'seq.7', 'seq.12']
    activation = torch.cuda.FloatTensor([0.])
    value_to_penalize = torch.cuda.FloatTensor([0.])

    def hook(m, i, o):
        nonlocal activation, value_to_penalize
        activation += o.sum() / BATCH_SIZE
        value_to_penalize += penalty_function(o)

    for name, module in myclass.named_modules():
        if name in hookable_modules:
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

        # Test network accuracy
        with torch.no_grad():
            myclass.eval()
            accuracy = []

            print(f"Epoch {epoch}, testing")
            for batch_id, sample in enumerate(tqdm(test_dataloader)):
                test_data, test_labels = sample
                test_data = test_data.to(device)
                test_labels = test_labels.to(device)

                outputs = myclass(test_data)
                accuracy.append(compute_accuracy(outputs, test_labels))

        print(f"loss={loss.item()}, accuracy_test={np.mean(accuracy)},",
              f"accuracy_train={np.mean(accuracy_train)}")

    # Save trained model
    if save:
        torch.save(myclass.state_dict(), 'models/' + save + '.pt')
        print(f"Model saved at {save}")

    activation_values.append(activation.item())
    test_acc.append(accuracy)
    target_losses.append(target_loss.item())


if __name__ == '__main__':
    N_EPOCHS = 5
    WEIGHT_DECAY = False

    # L2 neuron-wise
    penalties = np.logspace(-2.8, -2.2, 10)
    penalty_function = lambda out: 0.  # (out.mean(0)**2).sum()
    name = f"weightdecay_{N_EPOCHS}_epochs"
    # # L2 layer-wise
    # penalties = np.logspace(-6, 0, 10)
    # penalty_function = lambda out: (out.mean(0).sum())**2
    # name = f"L2_layerlevel_{N_EPOCHS}_epochs"
    # # L1 penalty
    # penalties = np.logspace(-5, -2, 10)
    # penalty_function = lambda out: out.mean(0).sum()
    # name = f"L1_sum_{N_EPOCHS}_epochs"

    test_acc = []
    target_losses = []
    activation_values = []

    for p in penalties:
        train(p, N_EPOCHS, save=name + '_' + str(p))
    results = np.asarray([penalties, activation_values,
                          test_acc, target_losses]).T
    np.savetxt('results/' + name + '.txt', results)
