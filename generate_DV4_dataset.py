from aer4manager import aedatconvert

conv = aedatconvert(
    data_list_file='./file_list.txt',
    accumulation_method='spikecount',
    accumulation_value=3000,
    test_fraction=0.2,
    output_resolution=(64, 64),
    crop_size=((45, 45), (2, 2)),
    hot_pixel_frequency=0.001,
).to_folder(
    out_dir='data',
    overwrite=False,
    compressed=False,  # if True smaller files, but slower to read and write
)
