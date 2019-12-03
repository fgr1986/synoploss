import torch
from torch import nn
from synoploss import NeuromorphicReLU, QuantizeLayer


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
            QuantizeLayer(quantize=quantize),
        )

    def forward(self, data):
        return self.seq(data)
