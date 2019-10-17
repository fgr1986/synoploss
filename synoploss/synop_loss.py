import torch
from synoploss import NeuromorphicReLU


class SynOpLoss(object):

    def __init__(self, modules):
        self.modules = []
        for module in modules:
            if isinstance(module, NeuromorphicReLU) and module.fanout > 0:
                self.modules.append(module)

        if len(self.modules) == 0:
            raise ValueError("No NeuromorphicReLU found in module list.")

    def __call__(self):
        device = self.modules[0].activity.device
        synops = torch.tensor([0.]).to(device)
        for module in self.modules:
            synops += module.activity
        return synops
