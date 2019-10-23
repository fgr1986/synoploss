import torch
from synoploss import NeuromorphicReLU


class SynOpLoss(object):

    def __init__(self, modules, sum_activations=True):
        self.modules = []
        for module in modules:
            if isinstance(module, NeuromorphicReLU) and module.fanout > 0:
                self.modules.append(module)

        if len(self.modules) == 0:
            raise ValueError("No NeuromorphicReLU found in module list.")

        self.sum_activations = sum_activations

    def __call__(self):
        synops = []
        for module in self.modules:
            synops.append(module.activity)
        if self.sum_activations:
            synops = torch.stack(synops).sum()
        return synops
