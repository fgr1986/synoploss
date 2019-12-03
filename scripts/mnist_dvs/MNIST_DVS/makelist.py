import glob
import numpy as np

SCALE = 16
TRAINSIZE = 800

file_labels_train, file_labels_test = [], []
for n in range(10):
    folder = f"grabbed_data{n}/scale{SCALE}/*"
    filelist = glob.glob(folder)
    labels = n * np.ones(len(filelist), dtype=np.int)
    file_label = np.vstack([filelist, labels])
    file_labels_train.append(file_label[:, :TRAINSIZE])
    file_labels_test.append(file_label[:, TRAINSIZE:])


file_labels_train = np.hstack(file_labels_train)
file_labels_test = np.hstack(file_labels_test)
print(file_labels_train.shape)
print(file_labels_test.shape)

np.savetxt("mnistdvs_filelist_train.txt", file_labels_train.T, delimiter='\t', fmt='%s')
np.savetxt("mnistdvs_filelist_test.txt", file_labels_test.T, delimiter='\t', fmt='%s')
