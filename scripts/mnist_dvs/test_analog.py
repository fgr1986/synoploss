from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as ttr
from torch.utils.data import DataLoader

from model import MNISTClassifier
from aer4manager import AERFolderDataset


def compute_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    acc = (predicted == target).sum().float() / len(target)
    return acc.cpu().numpy()


# Parameters
BATCH_SIZE = 256
QUANTIZE = True

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
    myclass = MNISTClassifier(quantize=QUANTIZE).to(device)

    # Load appropriate model
    state_dict = torch.load(path)

    # Do rescaling
    if w_rescale != 1.0:
        state_dict['seq.0.weight'] *= w_rescale

    myclass.load_state_dict(state_dict)

    # Set hooks
    activation = torch.cuda.FloatTensor([0.])
    weighed_activation = torch.cuda.FloatTensor([0.])

    def hook(m, i, o):
        nonlocal activation, weighed_activation
        activation += o.sum() / BATCH_SIZE
        weighed_activation += o.sum() * m.fanout / BATCH_SIZE

    for name, module in myclass.named_modules():
        if name in hookable_modules:
            module.fanout = fanouts[name]
            module.register_forward_hook(hook)

    # Test network accuracy
    with torch.no_grad():
        myclass.eval()
        accuracy = []

        for batch_id, sample in enumerate(tqdm(test_dataloader)):
            if batch_id > 50: break
            test_data, test_labels = sample
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            outputs = myclass(test_data)
            accuracy.append(compute_accuracy(outputs, test_labels))

        accuracy = np.mean(accuracy)
        print(f"accuracy_test={accuracy}")

    return path, activation.item(), weighed_activation.item(), accuracy


if __name__ == '__main__':
    import glob

    # Use this for the whole list of trained models
    # models_list = glob.glob('models/*')
    # results = list(map(test, models_list))

    # results = np.asarray(results)
    # print(results)
    # np.savetxt('results/analog_nonquant_tests.txt', results, fmt='%s')

    # Use this for a single model, but weight scaling
    model_path = "models/nopenalty_0.0.pt"
    scales = np.arange(0.1, 1.0, 0.1)
    results = [test(model_path, w_rescale) for w_rescale in scales]

    results = np.asarray(results)
    print(results)
    np.savetxt('results/weightscaled_quantized_tests.txt', results, fmt='%s')
