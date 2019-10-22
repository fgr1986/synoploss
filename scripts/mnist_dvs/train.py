from tqdm import tqdm
import numpy as np
import os

import torch
import torchvision.transforms as ttr
from torch.utils.data import DataLoader

from model import MNISTClassifier
from aer4manager import AERFolderDataset
from synoploss import SynOpLoss


def compute_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    acc = (predicted == target).sum().float() / len(target)
    return acc.cpu().numpy()


# Parameters
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('models', exist_ok=True)

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


def train(penalty_coefficient,
          epochs=10, save=False,
          quantize_training=False):
    print(f"Quantize training: {quantize_training}")
    # Define model and learning parameters
    classifier = MNISTClassifier(quantize=quantize_training).to(device)
    # Define loss
    criterion = torch.nn.CrossEntropyLoss()
    synops_criterion = SynOpLoss(classifier.modules())
    # Define optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    # Impose Kaiming He initialization
    for w in classifier.parameters():
        torch.nn.init.kaiming_uniform_(w, nonlinearity='relu')

    # Start training
    for epoch in range(epochs):
        classifier.train()
        accuracy_train = []

        print(f"Epoch {epoch}, training")
        for batch_id, sample in enumerate(tqdm(train_dataloader)):
            # if batch_id > 100: break
            optimizer.zero_grad()

            imgs, labels = sample
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = classifier(imgs)
            accuracy_train.append(compute_accuracy(outputs, labels))

            target_loss = criterion(outputs, labels)
            synops_loss = synops_criterion()
            loss = target_loss + penalty_coefficient * synops_loss

            loss.backward()
            optimizer.step()

        accuracy_train = np.mean(accuracy_train)
        print(f"loss={loss.item()}, accuracy_train={accuracy_train}")

    # Test network accuracy
    with torch.no_grad():
        classifier.eval()
        accuracy = []

        print(f"Epoch {epoch}, testing")
        for batch_id, sample in enumerate(tqdm(test_dataloader)):
            # if batch_id > 10: break
            test_data, test_labels = sample
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            outputs = classifier(test_data)
            activity = synops_criterion()
            accuracy.append(compute_accuracy(outputs, test_labels))

        accuracy = np.mean(accuracy)
        activity = activity.item()
        print(f"activity={activity}, accuracy_test={accuracy}")

    # Save trained model
    if save:
        savefile = f'models/{save}_{penalty_coefficient}.pth'
        torch.save(classifier.state_dict(), savefile)
        print(f"Model saved at {savefile}")

    with open("training_log.txt", "a") as f:
        f.write(f"{save} {penalty_coefficient} {epochs} {quantize_training} "
                f"{True} {activity} {accuracy} "  # True for backward compat
                f"{target_loss.item()} {savefile}\n")

    return penalty_coefficient, activity, accuracy, target_loss.item()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize_training', action='store_true',
                        default=False)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--n_models', type=int, default=10)
    opt = parser.parse_args()

    # no penalty
    name = "nopenalty" + ('-qtrain' if opt.quantize_training else '')
    train(
        0.0,
        epochs=opt.n_epochs,
        save=name,
        quantize_training=opt.quantize_training,
    )

    # L1 penalty with fanout weighing
    penalties = np.logspace(-9, -5, opt.n_models)
    name = "l1-fanout" + ('-qtrain' if opt.quantize_training else '')
    for p in penalties:
        train(
            p,
            epochs=opt.n_epochs,
            save=name,
            quantize_training=opt.quantize_training,
        )
