import torch
from synoploss import (
    NeuromorphicReLU,
    DynapSumPoolLayer,
    ScaledDroupout2d,
)

class CIFAR10AnalogueClassifier(torch.nn.Module):
    def __init__(self, quantize=False, dropout_rate=(0.2, 0.5), last_layer_relu=True):
        super().__init__()

        self.quantize = quantize
        self.last_layer_relu = last_layer_relu
        self.dropout_rate = dropout_rate
        layer_list = [torch.nn.Dropout2d(dropout_rate[0]),
            torch.nn.Conv2d(
                # ignore_synops=True,
                in_channels=3,
                out_channels=96,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            NeuromorphicReLU(quantize=quantize, fanout=3*3*96),
            torch.nn.Dropout2d(dropout_rate[1]),
            torch.nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            NeuromorphicReLU(quantize=quantize, fanout=3*3*192/2/2),
            torch.nn.Dropout2d(dropout_rate[1]),
            torch.nn.Conv2d(
                in_channels=96,
                out_channels=192,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(2, 2),
                bias=False,
            ),
            NeuromorphicReLU(quantize=quantize, fanout=3*3*192),
            torch.nn.Dropout2d(dropout_rate[1]),
            torch.nn.Conv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            NeuromorphicReLU(quantize=quantize, fanout=3*3*192),
            torch.nn.Dropout2d(dropout_rate[1]),
            torch.nn.Conv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            NeuromorphicReLU(quantize=quantize, fanout=3*3*192/2/2),
            torch.nn.Dropout2d(dropout_rate[1]),
            torch.nn.Conv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(2, 2),
                bias=False,
            ),
            NeuromorphicReLU(quantize=quantize, fanout=3*3*192),
            torch.nn.Dropout2d(dropout_rate[1]),
            torch.nn.Conv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            NeuromorphicReLU(quantize=quantize, fanout=1*1*192),
            torch.nn.Dropout2d(dropout_rate[1]),
            torch.nn.Conv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
                bias=False,
            ),
            NeuromorphicReLU(quantize=quantize, fanout=1*1*10),
            torch.nn.Dropout2d(dropout_rate[1]),
            torch.nn.Conv2d(
                in_channels=192,
                out_channels=10,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
                bias=False,
            ),]
        if last_layer_relu:
            layer_list.append(NeuromorphicReLU(quantize=quantize, fanout=1))
        layer_list.append(
            DynapSumPoolLayer(kernel_size=(8, 8), stride=(8, 8))
            # torch.nn.AvgPool2d(kernel_size=(6, 6), stride=(6, 6))
        )
        layer_list.append(
            torch.nn.Flatten()
        )
        self.seq = torch.nn.Sequential(*layer_list)
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            torch.nn.init.kaiming_normal_(w, nonlinearity="relu")
        # print("Init weights with kaiming")

    def forward(self, x):
        return self.seq(x)

