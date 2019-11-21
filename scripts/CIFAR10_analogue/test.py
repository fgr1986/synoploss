from train import *


def scale_first_weight(model, wscale=1):
    for i, w in enumerate(model.parameters()):
        if i < 1:
            w.data *= wscale
    return model


if __name__ == "__main__":
    device,train_dataloader,test_dataloader,spiking_test_dataloader,input_image_size = prepare()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=350)
    parser.add_argument("--n_times", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--b_save_model", type=str2bool, default=True)
    parser.add_argument("--str_log_file", type=str, default="log_inference_snn.txt")
    parser.add_argument("--target_scale", type=float, default=0.5)
    opt = parser.parse_args()

    print(opt)
    n_test = opt.n_test
    n_epochs = opt.n_epochs
    n_times = opt.n_times
    b_save_model = opt.b_save_model
    str_log_file = opt.str_log_file
    target_scale = opt.target_scale
    dropout_rate = (0.2, 0.1)

    classifier = CIFAR10AnalogueClassifier(quantize=True, dropout_rate=dropout_rate, last_layer_relu=True).to(device)
    for i_time in range(0, n_times):
        str_file_name = f"models/Nov19_d{dropout_rate[0]}_{dropout_rate[1]}_{i_time}.pth"

        print("---------start weight scale-----------------")
        for wscale in [0.5, 1, 1.8, 2, 2.5, 5]:
            if i_time == -1:
                wscale *= 100
            classifier.load_state_dict(torch.load(str_file_name))
            classifier = scale_first_weight(classifier, wscale=wscale)
            ann_accuracy, ann_synops = test(classifier, b_quantize=True)
            # snn_accuracy = ann_accuracy
            # snn_synops = ann_synops
            snn_accuracy, snn_synops = snn_test(classifier, n_dt=10, n_test=n_test)
            save_to_file(
                str_log_file, ann_accuracy, ann_synops, snn_accuracy, snn_synops, i_time
            )