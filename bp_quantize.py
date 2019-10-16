import torch
import torch.nn as nn

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



if __name__=="__main__":
    import numpy as np
    import torchvision
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader


    class MyModel(torch.nn.Module):
        def __init__(self, quantize=True):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(3, 3)),
                nn.ReLU(),
                QuantizeLayer(work=quantize),
                nn.AvgPool2d(kernel_size=(2, 2)),
                nn.Conv2d(8, 16, kernel_size=(3, 3)),
                nn.ReLU(),
                QuantizeLayer(work=quantize),
                nn.AvgPool2d(kernel_size=(2, 2)),
                nn.Conv2d(16, 32, kernel_size=(3, 3)),
                nn.ReLU(),
                QuantizeLayer(work=quantize),
                nn.AvgPool2d(kernel_size=(2, 2)),
                nn.Flatten(),
                nn.Linear(32, 10),
                QuantizeLayer(work=quantize)
            )

        def forward(self, data):
            return self.seq(data)


    model = MyModel(quantize=False).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


    # Load data
    mnist_dataset = MNIST('./data/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(mnist_dataset, batch_size=512, shuffle=True)

    n_epochs = 5

    for epoch in range(n_epochs):
        for data, label in dataloader:
            optimizer.zero_grad()
            out = model(data.cuda())
            loss = criterion(out, label.cuda())
            loss.backward()
            optimizer.step()
            print(loss.item())

    # Test model
    test_dataset= MNIST('./data/', train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    # Test model
    all_pred = []
    with torch.no_grad():
        model.eval()
        for data, label in test_dataloader:
            out = model(data.cuda())
            _, pred = out.max(1)
            all_pred.append((pred== label.cuda()).float().mean().item())

    print("Test accuracy original model: ", np.mean(all_pred))

    # Save the model params
    state_dict = model.state_dict()

    test_model = MyModel(quantize=True).cuda()
    test_model.load_state_dict(state_dict)
    all_pred = []
    with torch.no_grad():
        test_model.eval()
        for data, label in test_dataloader:
            out = test_model(data.cuda())
            _, pred = out.max(1)
            all_pred.append((pred== label.cuda()).float().mean().item())

    print("Test accuracy after quantize: ", np.mean(all_pred))


