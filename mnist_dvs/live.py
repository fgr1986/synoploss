from model import MNISTClassifier
from aermanager import LiveDv
import torch
from sinabs.from_torch import from_model

IMG_SIZE = 64
live = LiveDv(qlen=64, two_channel_mode=False, host='192.168.11.72')
model_analog = MNISTClassifier()
model_analog.load_state_dict(torch.load('models/l1-fanout-qtrain_1662224.4079924966.pth'))
in_shape = (1, 64, 64)
net = from_model(
    model_analog.seq,
    in_shape,
    threshold=1.0,
    membrane_subtract=1.0,
    threshold_low=-5.0,
).cuda()
net.spiking_model.eval()
adaptivepool = torch.nn.AdaptiveAvgPool2d((IMG_SIZE, IMG_SIZE))
factor = (256 / IMG_SIZE)**2


def transform(x):
    x = x[:, :, 2:-2, 45:-45]  # crop
    x = torch.tensor(x).float().cuda()
    x = adaptivepool(x) * factor
    return x


while True:
    batch = live.get_batch()
    batch = transform(batch)

    # net.reset_states()
    out = net(batch/3)
    maxval, pred_label = torch.max(out.sum(0), dim=0)

    THR = 30
    if maxval > THR:
        print(pred_label.item())
    else:
        print('.')
