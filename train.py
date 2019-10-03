from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as ttr
from torch.utils.data import DataLoader

from model import YundingClassifier
from aer4manager import AERFolderDataset


# Parameters
BATCH_SIZE = 128

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Prepare datasets and dataloaders
train_dataset = AERFolderDataset(
    root='data/',
    from_spiketrain=False,
    transform=ttr.Compose([
        ttr.ToPILImage(),
        ttr.RandomAffine(degrees=0, translate=(0, 0.2),
                         scale=(0.9, 1.1), fillcolor=0),
        ttr.RandomHorizontalFlip(),
        ttr.ToTensor()
    ]),
    which='train'
)

test_dataset = AERFolderDataset(
    root='data/',
    from_spiketrain=False,
    transform=ttr.Compose([
        ttr.ToPILImage(),
        ttr.ToTensor()
    ]),
    which='test'
)

print("Number of training frames:", len(train_dataset))
print("Number of testing frames:", len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def train(activation_penalty=0., epochs=10, save=False):
    # Define model and learning parameters
    myclass = YundingClassifier().to(device)
    # Define loss
    criterion = torch.nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = torch.optim.Adam(myclass.parameters(), lr=1e-3)
    # set hooks
    hookable_modules = ['seq.1', 'seq.4', 'seq.7', 'seq.12', 'seq.14']

    activation = torch.cuda.FloatTensor([0.])
    value_to_penalize = torch.cuda.FloatTensor([0.])

    def hook(m, i, o):
        nonlocal activation, value_to_penalize
        activation += o.sum() / BATCH_SIZE
        value_to_penalize += penalty_function(o)

    for name, module in myclass.named_modules():
        if name in hookable_modules:
            module.register_forward_hook(hook)
            print(name, module)

    # Impose Kaiming He initialization
    for w in myclass.parameters():
        torch.nn.init.kaiming_uniform_(w, nonlinearity='relu')

    # Start training
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        accuracy_train = []
        # Set to training mode
        myclass.train()
        running_loss = 0

        for batch_id, sample in enumerate(train_dataloader):
            imgs, labels = sample
            labels = labels.to(device)

            optimizer.zero_grad()
            activation = torch.cuda.FloatTensor([0.])
            value_to_penalize = torch.cuda.FloatTensor([0.])

            outputs = myclass(imgs.to(device))
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == labels.to(device)).sum().float() / len(labels)
            accuracy_train.append(acc.cpu().numpy())

            target_loss = criterion(outputs, labels)
            loss = target_loss + value_to_penalize * activation_penalty
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Test network accuracy
        with torch.no_grad():
            # Set to eval mode
            myclass.eval()
            accuracy = []
            for batch_id, sample in enumerate(test_dataloader):
                test_data, test_labels = sample

                outputs = myclass(test_data.to(device))
                _, predicted = torch.max(outputs, 1)
                acc = (predicted == test_labels.to(device)).sum().float() / len(test_labels)
                accuracy.append(acc.cpu().numpy())

        accuracy = np.mean(accuracy)
        accuracy_train = np.mean(accuracy_train)

        pbar.set_postfix(loss=running_loss, accuracy_test=accuracy,
                         accuracy_train=accuracy_train)

    # Save trained model
    if save:
        torch.save(myclass.state_dict(), 'models/' + save + '.pt')
        print(f"Model saved at {save}")

    activation_values.append(activation.item())
    test_acc.append(accuracy)
    target_losses.append(target_loss.item())


if __name__ == '__main__':
    N_EPOCHS = 30

    # L2 neuron-wise
    penalties = np.logspace(-3, 1, 10)
    penalty_function = lambda out: (out.mean(0)**2).sum()
    name = f"L2_neuronlevel_{N_EPOCHS}_epochs"
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
    np.savetxt(name + '.txt', results)
