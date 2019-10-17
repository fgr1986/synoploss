import torch
from aer4manager import AERFolderDataset
import torchvision.transforms as ttr
from torch.utils.data import DataLoader
from model import MNISTClassifier
from tqdm import tqdm
import numpy as np
import argparse
import uuid

"""
This file is to demonstrate the usefulness of quantization-aware
training. It trains a network either quantized or non quantized (see
the --quantize_training option). It then tests the network with and
without quantization.
"""

# ## LOAD ARGUMENTS ## #
parser = argparse.ArgumentParser()
parser.add_argument('--quantize_training', action='store_true', default=False)
opt = parser.parse_args()
print("Quantize training:", opt.quantize_training)

# ## PREPARE DATASET ## #
BATCH_SIZE = 512
EPOCHS = 10
NORM = 1.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ## DEFINE MODEL ## #
classifier = MNISTClassifier(quantize=opt.quantize_training).to(device)
# Define loss
criterion = torch.nn.CrossEntropyLoss()
# Define optimizer
optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)


# ## TRAINING ## #
for epoch in range(EPOCHS):
    classifier.train()

    print(f"Epoch {epoch}, training")
    for batch_id, sample in enumerate(tqdm(train_dataloader)):
        # if batch_id > 100: break
        optimizer.zero_grad()

        imgs, labels = sample
        imgs, labels = imgs.to(device), labels.to(device)
        imgs = imgs / NORM

        outputs = classifier(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Loss={loss.item()}, quantize_training={opt.quantize_training}")

hash = uuid.uuid4().hex[0:6]
path = f"quantization/{hash}.pth"
torch.save(classifier.state_dict(), path)


# ## TESTING ## #
def evaluate(state_dict, quantize_testing):
    with torch.no_grad():
        accuracy = []

        test_model = MNISTClassifier(quantize=quantize_testing).to(device)
        test_model.load_state_dict(state_dict)
        test_model.eval()

        print("** TESTING **")
        for batch_id, sample in enumerate(tqdm(test_dataloader)):
            # if batch_id > 10: break
            test_data, test_labels = sample
            test_data = test_data.to(device)
            test_data = test_data / NORM
            test_labels = test_labels.to(device)

            outputs = test_model(test_data)

            _, predicted = torch.max(outputs, 1)
            acc = (predicted == test_labels).sum().float() / len(test_labels)
            accuracy.append(acc.cpu().numpy())

        accuracy = np.mean(accuracy)

    print(f"accuracy_test={accuracy}, quantize_testing={quantize_testing}")
    return accuracy


acc_quantized = evaluate(classifier.state_dict(), True)
acc_analog = evaluate(classifier.state_dict(), False)

with open("quantization/results.txt", "a") as f:
    f.write(f"{hash}\t{opt.quantize_training}\tTrue\t{acc_quantized}\n")
    f.write(f"{hash}\t{opt.quantize_training}\tFalse\t{acc_analog}\n")
