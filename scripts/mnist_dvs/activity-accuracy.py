from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as ttr
from torch.utils.data import DataLoader
import argparse

from model import MNISTClassifier
from aer4manager import AERFolderDataset
from synoploss import SynOpLoss


def compute_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    acc = (predicted == target).sum().float() / len(target)
    return acc.cpu().numpy()


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

hookable_modules = ['seq.1', 'seq.4', 'seq.7', 'seq.12']
fanouts = {'seq.1': 72,
           'seq.4': 108,
           'seq.7': 432,
           'seq.12': 10,
           }


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
    activity_tracker = SynOpLoss(classifier.modules())

    # Test network accuracy
    with torch.no_grad():
        classifier.eval()
        accuracy, activity = [], []

        for batch_id, sample in enumerate(tqdm(test_dataloader)):
            if batch_id > opt.max_batches:
                break

            test_data, test_labels = sample
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            outputs = classifier(test_data)

            accuracy.append(compute_accuracy(outputs, test_labels))
            activity.append(activity_tracker().item())

        accuracy = np.mean(accuracy)
        activity = np.mean(activity)
        print(f"accuracy_test={accuracy}")

    return activity, accuracy


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    print("Quantized testing:", opt.quantize_testing)

    # # Get the whole list of trained models
    # f = np.loadtxt('training_log.txt', dtype=str).T
    # names, penalties, models = f[0], f[1].astype(np.float), f[-1]

    # # Select one kind of model, and test them all
    # chosen_name = "l1-fanout"
    # print(chosen_name)
    # idx = names == chosen_name
    # chosen_models = models[idx]
    # chosen_penalties = penalties[idx]

    # # check quantization during training
    # was_quantized_training = f[3][idx] == 'True'
    # assert all(~was_quantized_training)

    # # Go for testing
    # results = np.asarray([test(model) for model in chosen_models]).T
    # results = np.vstack([chosen_penalties, results[0], results[1]]).T
    # np.savetxt(f'results/{chosen_name}_qtest_{opt.quantize_testing}.txt',
    #            results, fmt='%s')

    # # Use this for a single model, but weight scaling
    # model_path = "models/nopenalty_0.0.pth"
    # scales = np.arange(0.1, 1.0, 0.05)
    # results = np.asarray([test(model_path, w_scale) for w_scale in scales]).T
    # results = np.vstack([scales, results[0], results[1]]).T
    # np.savetxt(f'results/weightscale_qtest_{opt.quantize_testing}.txt',
    #            results, fmt='%s')

    # Test the original model
    model_path = "models/nopenalty_0.0.pth"
    results = test(model_path, 1.0)
    results = np.asarray([[0.0, results[0], results[1]]])
    np.savetxt(f'results/nopenalty_qtest_{opt.quantize_testing}.txt',
               results, fmt='%s')
