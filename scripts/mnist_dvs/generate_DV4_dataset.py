from aer4manager import aedatconvert

root = '/home/martino/StorageRoom/datasets/MNIST_DVS'
for fraction in ['train', 'test']:

    aedatconvert(
        data_list_file=root+f'/mnistdvs_filelist_{fraction}.txt',
        accumulation_method='spikecount',
        accumulation_value=3000,
        test_fraction=0.,
        output_resolution=(64, 64),
        crop_size=((0, 0), (0, 0)),
        hot_pixel_frequency=0.0,
        root=root,
        dvs_model="DVS128",
    ).to_folder(
        out_dir=f'data/{fraction}',
        overwrite=False,
        compressed=True,  # if True smaller files, but slower to read and write
    )