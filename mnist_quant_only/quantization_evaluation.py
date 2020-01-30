import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
from model import MyModel


mnist_dataset = MNIST('./data/', train=True, download=True,
                      transform=transforms.ToTensor())
dataloader = DataLoader(mnist_dataset, batch_size=128, shuffle=True)

# Test model
test_dataset = MNIST('./data/', train=False, download=True,
                     transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)


QUANTIZE_TRAIN = True
scales = np.arange(0.1, 5.1, 0.2)
N_EPOCHS = 50


results = []
print('Quantize training', QUANTIZE_TRAIN)
for scale in scales:

    model = MyModel(quantize=QUANTIZE_TRAIN).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                gamma=0.316)

    previous_loss = 0.00

    # Train model
    for epoch in range(N_EPOCHS):
        for data, label in dataloader:
            optimizer.zero_grad()
            data = scale * data.cuda()
            out = model(data)
            loss = criterion(out, label.cuda())
            loss.backward()
            optimizer.step()

        print(loss.item())

        if loss.item() == previous_loss:
            print("Converged, breaking training")
            break
        previous_loss = loss.item()
        scheduler.step()

    # Save the model params
    state_dict = model.state_dict()
    torch.save(state_dict, f'models/scale_{scale}_qtrain_{QUANTIZE_TRAIN}.pth')

    test_model_q = MyModel(quantize=True).cuda()
    test_model_a = MyModel(quantize=False).cuda()
    test_model_q.load_state_dict(state_dict)
    test_model_a.load_state_dict(state_dict)

    # spikecounter = SynOpCounter(test_model_q.modules())

    # Test
    all_pred_q = []
    all_pred_a = []
    all_counts = []
    with torch.no_grad():
        test_model_q.eval()
        test_model_a.eval()

        for data, label in test_dataloader:
            data = scale * data.cuda()

            out_q = test_model_q(data)
            out_a = test_model_a(data)

            _, pred_q = out_q.max(1)
            _, pred_a = out_a.max(1)

            all_pred_q.append((pred_q == label.cuda()).float().mean().item())
            all_pred_a.append((pred_a == label.cuda()).float().mean().item())
            # all_counts.append(spikecounter().cpu().numpy())

    print("Test accuracy after quantize: ", np.mean(all_pred_q))
    print("Test accuracy not quantized: ", np.mean(all_pred_a))

    results.append([scale, np.mean(all_pred_q), np.mean(all_pred_a)])

np.savetxt(f'res_both_qtrain_{QUANTIZE_TRAIN}.txt', results)
