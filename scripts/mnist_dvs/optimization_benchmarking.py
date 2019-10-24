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


def compute_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    acc = (predicted == target).sum().float() / len(target)
    return acc.cpu().numpy()


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
        accuracy = []

        for batch_id, sample in enumerate(tqdm(test_dataloader)):
            if batch_id > opt.max_batches:
                break

            test_data, test_labels = sample
            test_data = test_data.to(device)

            output = classifier(test_data)
            accuracy.append(compute_accuracy(output, test_labels.to(device)))

            activity.append(activity_tracker())

    return np.mean(detach(activity), axis=0), np.mean(accuracy)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # test non-optimized model
    baseline_activity, baseline_accuracy = test('models/nopenalty_0.0.pth')

    # test optimized model
    optimized_activity, optimized_accuracy = test(
        'models/l1-fanout-qtrain_462932.04169587774.pth')

    # plot layer by layer comparison
    layers = ['ReLU1', 'ReLU2', 'ReLU3']
    fig, ax = plt.subplots(1, 2, figsize=(8, 4),
                           gridspec_kw={'width_ratios': [2.5, 1]})
    xticks = np.arange(len(layers))
    ax[0].bar(xticks - 0.2, baseline_activity / 1e6, width=0.4, align='center',
              label="Baseline", color='C3')
    ax[0].bar(xticks + 0.2, optimized_activity / 1e6, width=0.4,
              align='center', label=r"SynOp + Quant., selected model", color='C0')
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(layers)
    ax[0].set_ylabel(r"Estimated synaptic operations ($\times 10^6$)")
    ax[0].legend(loc="upper right")

    ax[1].bar([-0.2], baseline_accuracy, color='C3', width=0.4, align='center')
    ax[1].bar([+0.2], optimized_accuracy, color='C0', width=0.4, align='center')
    ax[1].set_xticks([0])
    ax[1].set_xticklabels(['Accuracy'])
    ax[1].set_xlim([-0.6, 0.6])
    # ax.set_title("Synaptic operations per layer, optimized model vs. baseline")
    plt.show()
    fig.savefig('figures/compare.pdf')
