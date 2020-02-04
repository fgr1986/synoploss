# This is for testing models with the 'smart' rescaling proposed by Rueckauer 2017,
# where the gain for each layer is estimated and used to normalize.
from torch.utils.data import DataLoader
from aermanager import AERFolderDataset
import torchvision.transforms as ttr
from model import MNISTClassifier
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

outputs_register = {}


# Create a function that creates hook functions
def hook_factory(name):
    def hook(self, input, output):
        if name not in outputs_register.keys():
            outputs_register[name] = []
        outputs_register[name].append(output.data)
    return hook


# Prepare datasets and dataloaders
train_dataset = AERFolderDataset(
    root='data/train/',
    from_spiketrain=False,
    transform=ttr.ToTensor(),
)

BATCH_SIZE = 256
print("Number of training frames:", len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# load the model
model_path = "models/nopenalty_0.0.pth"
state_dict = torch.load(model_path)
model = MNISTClassifier()
model.load_state_dict(state_dict)
model.eval().cuda()

hookable_layers = ['input_relu', 'seq.1', 'seq.4', 'seq.7', 'seq.12']
for lname, layer in model.named_modules():
    if lname in hookable_layers:
        layer.register_forward_hook(hook_factory(lname))

for batch_id, sample in enumerate(tqdm(train_dataloader)):
    data, label = sample
    model(data.cuda())

    if batch_id > 100:
        break  # we don't have enough memory to save all outputs


scales = {}
for i, (layer, output) in enumerate(outputs_register.items()):
    layer_outputs = torch.stack(output).cpu().numpy().ravel()
    scales[layer] = np.percentile(layer_outputs, q=99.99)

#     plt.subplot(2, 3, i+1)
#     plt.hist(layer_outputs, bins=50)
#     plt.yscale('log')
#     plt.axvline(scales[layer])
# plt.show()

state_dict['seq.0.weight'] *= scales['input_relu'] / scales['seq.1']
state_dict['seq.3.weight'] *= scales['seq.1'] / scales['seq.4']
state_dict['seq.6.weight'] *= scales['seq.4'] / scales['seq.7']
state_dict['seq.11.weight'] *= scales['seq.7'] / scales['seq.12']

torch.save(state_dict, 'models/nopenalty_renormalized_gain.pth')
