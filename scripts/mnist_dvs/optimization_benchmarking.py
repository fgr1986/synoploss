from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as ttr
from torch.utils.data import DataLoader
import argparse

from model import MNISTClassifier
from aer4manager import AERFolderDataset
from synoploss import SynOpLoss

from test_spiking import test_spiking


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
    # test non-optimized model
    baseline_activity, baseline_accuracy = test_spiking(
        'models/nopenalty_0.0.pth', return_all_synops=True
    )

    # test optimized model
    optimized_activity, optimized_accuracy = test_spiking(
        'models/l1-fanout-qtrain_321289.514081772.pth',
        return_all_synops=True
    )

    baseline_activity = baseline_activity[baseline_activity > 0]
    optimized_activity = optimized_activity[optimized_activity > 0]

    np.savez(
        'opt_benchmark.npz',
        baseline_activity=baseline_activity,
        optimized_activity=optimized_activity,
        baseline_accuracy=baseline_accuracy,
        optimized_accuracy=optimized_accuracy
    )
