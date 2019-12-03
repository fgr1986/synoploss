# SynOpLoss

Repository for code and examples related to quantization-aware training of spiking convolutional neural networks,
and SynOp loss for lowering power consumption. This is the code that was used to obtain the results of the following paper:

Martino Sorbaro, Qian Liu, Massimo Bortone, and Sadique Sheik.
Optimizing the energy consumption of spiking neural networks for neuromorphic applications. (2019)

Please cite that paper if you use this code in an academic publication.

## Contents

- `synoploss/`: contains code for quantizing networks and for computing the SynOp count
- `tests/`: contains tests for the previous code
- `scripts/`: contains the training and simulations presented in the paper

## Reproducing the mnist-dvs training

The code necessary for this purpose is in the `scripts/mnist_dvs` folder.

1. First, download the dataset from [the Seville microelectronic institute's website](http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html)
and unzip them in the MNIST_DVS folder.

2. Our code trains networks on 3000-spikes accumulated frames, but preserves the corresponding raw spike trains for use at higher time 
resolution for testing on spiking networks. To generate data in the right format, you can use the file `generate_DV4_dataset.py`.
For this, you will also need the `aer4manager` software, available in a separate repository.

3. You can now train the models using the `train.py` file. This will train many models, with a range of target values for the SynOp loss.
Add the `--quantize_training` flag if you would like to add quantization-aware training. Models are saved in the `models/` folder.
Training-time estimates on activity and accuracy, loss, corresponding saved file, and various hyperparameters are saved in `training_log.txt`.

4. You can test these models on a Sinabs spiking network using `test_spiking.py`. You will need to install [Sinabs](gitlab.com/aiCTX/sinabs) for this. You must use the `--mode` flag, specifying 'quantized' (tests all the models trained with quantization), 'nonquantized' (tests the models without quantization at training time), 'nopenalty' (tests the single model called nopenalty, the one without synoploss) or 'weightscaling' (tests the nopenalty model multiple times with weight rescaling).

5. `optimization_benchmarking.py` can be optionally used for more detailed testing of a single model.

6. Use the `GeneratePlots.ipynb` Jupyter notebook to create the figures in the paper.
