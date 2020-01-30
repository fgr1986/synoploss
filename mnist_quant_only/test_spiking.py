import torch
from sinabs.from_torch import from_model
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from model import MyModel


QUANTIZE_TRAIN = True
N_TIMESTEPS = 100
scale_up = 1


def test_spiking(inp_rescale=1.0, return_all_synops=False):
    path_to_weights = f'models/scale_{inp_rescale}_qtrain_{QUANTIZE_TRAIN}.pth'
    # instantiate dataloader
    test_dataset = MNIST('./data/', train=False, download=True,
                         transform=transforms.ToTensor())

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True
    )

    # load the model
    state_dict = torch.load(path_to_weights)

    # # Do rescaling
    # if w_rescale != 1.0:
    #     state_dict['seq.0.weight'] *= w_rescale

    model = MyModel()
    model.load_state_dict(state_dict)

    in_shape = (1, 64, 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = from_model(
        model.seq,
        in_shape,
        threshold=1.0,
        membrane_subtract=1.0,
        threshold_low=None,
        bias_rescaling=N_TIMESTEPS,
    ).to(device)
    net.spiking_model.eval()

    accuracy = []
    synops = []

    with torch.no_grad():
        # loop over the input files
        for i, sample in enumerate(tqdm(test_dataloader)):
            if i > 5000: break
            test_data, test_labels = sample
            input_frames = test_data[0].to(device)
            input_frames = input_frames.unsqueeze(0) * inp_rescale
            input_frames = input_frames.expand((N_TIMESTEPS, -1, -1, -1))
            input_frames = input_frames / N_TIMESTEPS * scale_up

            # we reset the network when changing file
            net.reset_states()

            # loop over the 1 ms frames WITHIN a single input file
            outputs = net.spiking_model(input_frames)

            synops_df = net.get_synops(0)

            if return_all_synops:
                synops.append(synops_df['SynOps'])
            else:
                synops.append(synops_df['SynOps'].sum())

            _, predicted = outputs.sum(0).max(0)
            correctness = (predicted == test_labels.to(device))
            accuracy.append(correctness.cpu().numpy())

    return inp_rescale * scale_up, np.mean(synops, axis=0), np.mean(accuracy)


if __name__ == '__main__':
    from multiprocessing import Pool
    import os
    os.makedirs('results', exist_ok=True)
    P = Pool(4)

    scales = np.arange(0.1, 5.1, 0.2)

    # Get the whole list of trained models
    results = P.map(test_spiking, scales)
    np.savetxt(f'results/sinabs_results_qtrain_{QUANTIZE_TRAIN}.txt',
               results, fmt='%s')
