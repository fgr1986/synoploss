import torch


class MNISTClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8,
                            kernel_size=(3, 3), bias=False),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=8, out_channels=12,
                            kernel_size=(3, 3), bias=False),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=12, out_channels=12,
                            kernel_size=(3, 3), bias=False),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Dropout2d(0.5),
            torch.nn.Flatten(),
            torch.nn.Linear(432, 10, bias=False),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(x)
