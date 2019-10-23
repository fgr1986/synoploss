import torch
from synoploss import NeuromorphicReLU


class MNISTClassifier(torch.nn.Module):
    def __init__(self, quantize=False):
        super().__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8,
                            kernel_size=(3, 3), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=108),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=8, out_channels=12,
                            kernel_size=(3, 3), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=108),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=12, out_channels=12,
                            kernel_size=(3, 3), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=10),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Dropout2d(0.5),
            torch.nn.Flatten(),
            torch.nn.Linear(432, 10, bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=0),
        )

    def forward(self, x):
        return self.seq(x)
