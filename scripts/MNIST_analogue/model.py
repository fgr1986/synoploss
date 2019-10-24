import torch
from synoploss import NeuromorphicReLU


class MNISTAnalogueClassifier(torch.nn.Module):
    def __init__(self, quantize=False):
        super().__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16,
                            kernel_size=(5, 5), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=16*5*5),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=16, out_channels=64,
                            kernel_size=(5, 5), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=64*5*5),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Dropout2d(0.5),
            torch.nn.Conv2d(in_channels=64, out_channels=10,
                            kernel_size=(4, 4), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=10*4*4),
            torch.nn.Flatten(),
        )
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            torch.nn.init.kaiming_uniform_(w, nonlinearity='relu')
        print("Init weights with kaiming")

    def forward(self, x):
        return self.seq(x)


class BodoClassifier(torch.nn.Module):
    def __init__(self, quantize=False):
        super().__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16,
                            kernel_size=(5, 5), stride=(2, 2), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=16*3*3),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=16, out_channels=32,
                            kernel_size=(3, 3), padding=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=32*3*3),
            torch.nn.Conv2d(in_channels=32, out_channels=8,
                            kernel_size=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=8*1*1),
            torch.nn.Conv2d(in_channels=8, out_channels=10,
                            kernel_size=(3, 3), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=10*3*3),
            torch.nn.Dropout2d(0.5),
            torch.nn.Conv2d(in_channels=10, out_channels=10,
                            kernel_size=(4, 4), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=10*4*4),
            torch.nn.Flatten(),
        )
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            torch.nn.init.kaiming_uniform_(w, nonlinearity='relu')
        print("Init weights with kaiming")

    def forward(self, x):
        return self.seq(x)

