import torch
from sinabs.from_torch import from_model
from model import MNISTClassifier
from aermanager import AERFolderDataset
import numpy as np
from tqdm import tqdm


def test_spiking(path_to_weights, w_rescale=1.0, return_all_synops=False):
    # instantiate dataloader
    test_dataset = AERFolderDataset(
        root='data/test',
        from_spiketrain=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True
    )

    # load the model
    state_dict = torch.load(path_to_weights)

    # Do rescaling
    if w_rescale != 1.0:
        state_dict['seq.0.weight'] *= w_rescale

    model = MNISTClassifier()
    model.load_state_dict(state_dict)

    in_shape = (1, 64, 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = from_model(
        model.seq,
        in_shape,
        threshold=1.0,
        membrane_subtract=1.0,
        threshold_low=None
    ).to(device)
    net.spiking_model.eval()

    accuracy = []
    synops = []

    with torch.no_grad():
        # loop over the input files
        for i, sample in enumerate(tqdm(test_dataloader)):
            # if i > 2000: break
            test_data, test_labels = sample
            input_frames = test_data[0].to(device)
            input_frames = input_frames.unsqueeze(1)

            # we reset the network when changing file
            net.reset_states()

            # loop over the 1 ms frames WITHIN a single input file
            outputs = net.spiking_model(input_frames)

            synops_df = net.get_synops(3000)

            if return_all_synops:
                synops.append(synops_df['SynOps'])
            else:
                synops.append(synops_df['SynOps'].sum())

            _, predicted = outputs.sum(0).max(0)
            correctness = (predicted == test_labels.to(device))
            accuracy.append(correctness.cpu().numpy())

    return np.mean(synops, axis=0), np.mean(accuracy)


if __name__ == '__main__':
    from multiprocessing import Pool
    import os
    import argparse
    os.makedirs('results', exist_ok=True)
    P = Pool(4)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type='string', default=False,
        help='one of "nonquantized", "quantized", "nopenalty", "weightscale"')
    opt = parser.parse_args()

    # Get the whole list of trained models
    f = np.loadtxt('training_log.txt', dtype=str).T
    names, penalties, models = f[0], f[1].astype(np.float), f[-1]

    if opt.mode == 'quantized':
        # Select one kind of model, and test them all
        chosen_name = "l1-fanout-qtrain"
        print(chosen_name)
        idx = names == chosen_name
        chosen_models = models[idx]
        chosen_penalties = penalties[idx]

        # check quantization during training
        was_quantized_training = f[3][idx] == 'True'
        assert all(was_quantized_training)

        # Go for testing
        results = np.asarray(P.map(test_spiking, chosen_models)).T
        results = np.vstack([chosen_penalties, results[0], results[1]]).T
        np.savetxt(f'results/{chosen_name}_spiking_nothr.txt',
                   results, fmt='%s')

    elif opt.mode == 'nonquantized':
        # Select one kind of model, and test them all
        chosen_name = "l1-fanout"
        print(chosen_name)
        idx = names == chosen_name
        chosen_models = models[idx]
        chosen_penalties = penalties[idx]

        # check quantization during training
        was_quantized_training = f[3][idx] == 'True'
        assert all(~was_quantized_training)

        # Go for testing
        results = np.asarray(P.map(test_spiking, chosen_models)).T
        results = np.vstack([chosen_penalties, results[0], results[1]]).T
        np.savetxt(f'results/{chosen_name}_spiking_nothr.txt',
                   results, fmt='%s')

    elif opt.mode == 'weightscale':
        # Use this for a single model, but weight scaling
        model_path = "models/nopenalty_0.0.pth"
        scales = np.arange(0.1, 1.0, 0.05)
        results = np.asarray([test_spiking(model_path, w_scale) for w_scale in scales]).T
        results = np.vstack([scales, results[0], results[1]]).T
        np.savetxt(f'results/weightscale_spiking_nothr.txt',
                   results, fmt='%s')

    elif opt.mode == 'nopenalty':
        # Test the original model
        model_path = "models/nopenalty_0.0.pth"
        results = test_spiking(model_path, 1.0)
        results = np.asarray([[0.0, results[0], results[1]]])
        np.savetxt(f'results/nopenalty_spiking_nothr.txt',
                   results, fmt='%s')

    else:
        raise ValueError("Unknown mode")
