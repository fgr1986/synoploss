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


class QuantizedSurrogateReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.nn.functional.relu(input).floor()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (torch.sign(input) + 1.0) / 2.0


class NeuromorphicReLU(torch.nn.Module):
    def __init__(self, quantize=True, fanout=1):
        super().__init__()
        self.quantize = quantize
        self.fanout = fanout

    def forward(self, input):
        if self.quantize:
            output = QuantizedSurrogateReLUFunction.apply(input)
        else:
            output = torch.nn.functional.relu(input)
        self.activity = output.sum() / len(output) * self.fanout
        return output
