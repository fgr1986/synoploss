import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
from torch import nn
from synoploss import NeuromorphicReLU, SynOpLoss


class MyModel(torch.nn.Module):
    def __init__(self, quantize=True):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3)),
            NeuromorphicReLU(quantize=quantize, fanout=1),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(8, 16, kernel_size=(3, 3)),
            NeuromorphicReLU(quantize=quantize, fanout=1),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            NeuromorphicReLU(quantize=quantize, fanout=1),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(32, 10),
            NeuromorphicReLU(quantize=quantize, fanout=1),
        )

    def forward(self, data):
        return self.seq(data)


mnist_dataset = MNIST('./data/', train=True, download=True,
                      transform=transforms.ToTensor())
dataloader = DataLoader(mnist_dataset, batch_size=512, shuffle=True)

# Test model
test_dataset = MNIST('./data/', train=False, download=True,
                     transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)


QUANTIZE_TRAIN = True
QUANTIZE_TEST = True
scales = np.arange(0.1, 3.1, 0.1)
N_EPOCHS = 40


results = []
print('Quantize training', QUANTIZE_TRAIN)
print('Quantize test', QUANTIZE_TEST)
for scale in scales:

    model = MyModel(quantize=QUANTIZE_TRAIN).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    spikecounter = SynOpLoss(model.modules())

    # Train model
    for epoch in range(N_EPOCHS):
        for data, label in dataloader:
            optimizer.zero_grad()
            data = scale * data.cuda()
            out = model(data)
            loss = criterion(out, label.cuda())
            loss.backward()
            optimizer.step()
        print(loss.item())

    # Save the model params
    state_dict = model.state_dict()
    test_model = MyModel(quantize=QUANTIZE_TEST).cuda()
    test_model.load_state_dict(state_dict)

    # Test
    all_pred = []
    all_counts = []
    with torch.no_grad():
        test_model.eval()
        for data, label in test_dataloader:
            data = scale * data.cuda()
            out = test_model(data)
            _, pred = out.max(1)
            all_pred.append((pred == label.cuda()).float().mean().item())
            all_counts.append(spikecounter().cpu().numpy())

    print("Test accuracy after quantize: ", np.mean(all_pred))
    results.append([scale, np.mean(all_pred), np.mean(all_counts)])

np.savetxt(f'res2_qtrain_{QUANTIZE_TRAIN}_qtest_{QUANTIZE_TEST}.txt', results)
