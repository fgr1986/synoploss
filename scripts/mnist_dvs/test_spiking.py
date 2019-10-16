import torch
from sinabs.from_torch import from_model
from model import MNISTClassifier
from aer4manager import AERFolderDataset
import numpy as np
from tqdm import tqdm


def test_spiking(path_to_weights, threshold=1.0):
    # instantiate dataloader
    test_dataset = AERFolderDataset(
        root='data/test',
        from_spiketrain=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True
    )
    # load the model
    weight_dict = torch.load(path_to_weights)
    model = MNISTClassifier()
    model.load_state_dict(weight_dict)

    in_shape = (1, 64, 64)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = from_model(
        model,
        in_shape,
        threshold=threshold,
        membrane_subtract=threshold,
        threshold_low=-1.0
    ).to(device)
    net.spiking_model.eval()

    accuracy = []
    synops = []

    with torch.no_grad():
        # loop over the input files
        for i, sample in enumerate(tqdm(test_dataloader)):
            if i > 10000: break
            test_data, test_labels = sample
            input_frames = test_data[0].to(device)
            input_frames = input_frames.unsqueeze(1)

            # we reset the network when changing file
            net.reset_states()

            # loop over the 1 ms frames WITHIN a single input file
            outputs = net.spiking_model(input_frames)

            synops_df = net.get_synops(3000)
            synops.append(synops_df['SynOps/s'].sum())

            _, predicted = outputs.sum(0).max(0)
            correctness = (predicted == test_labels.to(device))
            accuracy.append(correctness.cpu().numpy())

    return np.mean(accuracy), np.mean(synops)


def test_varying_thres(threshold):
    model = 'models/nopenalty_0.0.pt'
    return test_spiking(model, threshold=threshold)


if __name__ == '__main__':
    import glob
    from multiprocessing import Pool

    # Use this for the whole list of trained models
    models_list = glob.glob('models/l1fanout*')
    P = Pool(10)
    results = P.map(test_spiking, models_list)
    P.close()
    results = np.array(results).T
    results = np.vstack([models_list, results[0], results[1]]).T

    # # Use below for variable threshold
    # thresholds = np.arange(1, 4, 0.3)
    # thr_descriptor = [f'nopen_thres_30_epochs_{thr}' for thr in thresholds]

    # P = Pool(5)
    # results = P.map(test_varying_thres, thresholds)
    # P.close()
    # results = np.array(results).T
    # results = np.vstack([thr_descriptor, results[0], results[1]]).T

    # Leave this there.
    np.savetxt('results/spk_results_fanout.txt', results, fmt='%s')
