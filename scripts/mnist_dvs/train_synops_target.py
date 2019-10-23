from train import train
import numpy as np

alphas, orig_synops, orig_acc = np.loadtxt(
    'results/l1-fanout_qtest_True.txt').T

N_EPOCHS = 8
QUANTIZED_TRAINING = True

# TRAINING
target_synops = orig_synops[::4]
results = []
name = "targetloss" + ('-qtrain' if QUANTIZED_TRAINING else '')
for p in target_synops:
    target, activity, accuracy, target_loss = train(
        p,
        epochs=N_EPOCHS,
        save=name,
        quantize_training=QUANTIZED_TRAINING,
        as_target_synops=True,
    )

    results.append([target, activity, accuracy])

np.savetxt(f'results/{name}.txt', results)
