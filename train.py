from tqdm import tqdm
from numpy import mean

import torch
import torchvision.transforms as ttr
from torch.utils.data import DataLoader

from model import YundingClassifier
from aer4manager import AERFolderDataset


# Parameters
MODEL_SAVEPATH = './testing_model'
N_EPOCHS = 10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define model and learning parameters
myclass = YundingClassifier().to(device)
# Impose Kaiming He initialization
for w in myclass.parameters():
    torch.nn.init.kaiming_uniform_(w, nonlinearity='relu')
# Define loss
criterion = torch.nn.CrossEntropyLoss()
# Define optimizer
optimizer = torch.optim.Adam(myclass.parameters(), lr=1e-3)

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

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# Start training
pbar = tqdm(range(N_EPOCHS))

for epoch in pbar:
    accuracy_train = []
    # Set to training mode
    myclass.train()
    running_loss = 0

    for batch_id, sample in enumerate(train_dataloader):
        imgs, labels = sample
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = myclass(imgs.to(device))

        _, predicted = torch.max(outputs, 1)
        binary_labels = labels  # torch.max(labels, 1)
        acc = (predicted == binary_labels.to(device)).sum().float() / len(labels)
        accuracy_train.append(acc.cpu().numpy())

        loss = criterion(outputs, labels)
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
            test_bin_labels = test_labels  # torch.max(test_labels, 1)

            acc = (predicted == test_bin_labels.to(device)).sum().float() / len(test_labels)
            accuracy.append(acc.cpu().numpy())
        accuracy = mean(accuracy)
    accuracy_train = mean(accuracy_train)

    pbar.set_postfix(loss=running_loss, accuracy_test=accuracy,
                     accuracy_train=accuracy_train)

# Save trained model
torch.save(myclass.state_dict(), MODEL_SAVEPATH + '.pt')
print(f"Model saved at {MODEL_SAVEPATH}")
