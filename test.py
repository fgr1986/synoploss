import torch
from sinabs.from_torch import from_model
from model import YundingClassifier
from aer4manager import AERFolderDataset
import numpy as np
from tqdm import tqdm


def test_spiking(path_to_weights):
    # instantiate dataloader
    test_dataset = AERFolderDataset(
        root='data/',
        from_spiketrain=True,
        which='test',
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True
    )
    # load the model
    weight_dict = torch.load(path_to_weights)
    model = YundingClassifier()
    model.load_state_dict(weight_dict)

    in_shape = (1, 64, 64)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = from_model(model, in_shape).to(device)
    net.spiking_model.eval()

    accuracy = []
    synops = []

    with torch.no_grad():
        # loop over the input files
        for i, sample in enumerate(tqdm(test_dataloader)):
            # if i > 50: break
            test_data, test_labels = sample
            input_frames = test_data[0].to(device)

            # we reset the network when changing file
            net.reset_states()

            # loop over the 1 ms frames WITHIN a single input file
            outputs = torch.cuda.FloatTensor([[0., 0.]])
            for frame in input_frames:
                frame = frame.unsqueeze(0).unsqueeze(0)
                outputs += net.spiking_model(frame)

                synops_df = net.get_synops(1)
                synops.append(synops_df['SynOps'].mean())

            _, predicted = torch.max(outputs, 1)
            correctness = (predicted == test_labels.to(device))
            accuracy.append(correctness.cpu().numpy())

    return np.mean(accuracy), np.mean(synops)


if __name__ == '__main__':
    import glob
    from multiprocessing import Pool

    models_list = glob.glob('models/*')

    P = Pool(4)
    results = P.map(test_spiking, models_list)
    P.close()
    results = np.array(results).T
    results = np.vstack([models_list, results[0], results[1]]).T
    np.savetxt('spk_results.txt', results, fmt='%s')
