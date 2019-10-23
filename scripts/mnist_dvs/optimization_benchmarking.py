from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as ttr
from torch.utils.data import DataLoader
import argparse

from model import MNISTClassifier
from aer4manager import AERFolderDataset
from synoploss import SynOpLoss


# Parameters
BATCH_SIZE = 256

parser = argparse.ArgumentParser()
parser.add_argument('--quantize_testing', action='store_true', default=False)
parser.add_argument('--max_batches', type=int, default=1000000)
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare dataset and dataloader
test_dataset = AERFolderDataset(
    root='data/test/',
    from_spiketrain=False,
    transform=ttr.ToTensor(),
)

print("Number of testing frames:", len(test_dataset))

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


def detach(activity):
    for activations in activity:
        for (i, activation) in enumerate(activations):
            activations[i] = activation.item()
    return np.array(activity)


def test(path, w_rescale=1.0):
    # Define model and learning parameters
    classifier = MNISTClassifier(quantize=opt.quantize_testing).to(device)

    # Load appropriate model
    state_dict = torch.load(path)

    # Do rescaling
    if w_rescale != 1.0:
        state_dict['seq.0.weight'] *= w_rescale

    classifier.load_state_dict(state_dict)

    # Set hooks
    activity_tracker = SynOpLoss(classifier.modules(), sum_activations=False)

    # Test network accuracy
    with torch.no_grad():
        classifier.eval()
        activity = []

        for batch_id, sample in enumerate(tqdm(test_dataloader)):
            if batch_id > opt.max_batches:
                break

            test_data, test_labels = sample
            test_data = test_data.to(device)

            _ = classifier(test_data)

            activity.append(activity_tracker())

    return detach(activity)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # test non-optimized model
    baseline_activity = test('models/nopenalty_0.0.pth')

    # test optimized model
    optimized_activity = test('models/l1-fanout-qtrain_4.0842386526745176e-07.pth')

    # compute reduction in activity
    baseline_activity = np.mean(baseline_activity, axis=0)
    optimized_activity = np.mean(optimized_activity, axis=0)
    total_reduction = np.sum(np.abs(baseline_activity - optimized_activity)) / np.sum(baseline_activity)

    # plot layer by layer comparison
    layers = ['ReLU1', 'ReLU2', 'ReLU3', 'ReLU4']
    fig, ax = plt.subplots()
    xticks = np.arange(len(layers))
    ax.bar(xticks - 0.2, baseline_activity / 1e6, width=0.4, align='center',
           label="Baseline", color='C3')
    ax.bar(xticks + 0.2, optimized_activity / 1e6, width=0.4,
           align='center', label=r"SynOp + Quant., $\alpha = 4.08\times 10^{-7}$", color='C0')
    ax.set_xticks(xticks)
    ax.set_xticklabels(layers)
    ax.set_ylabel(r"Estimated synaptic operations ($\times 10^6$)")
    ax.legend(loc="upper right")
    ax.set_title("Synaptic operations per layer, optimized model vs. baseline")
    plt.show()
    fig.savefig('figures/compare.pdf')
