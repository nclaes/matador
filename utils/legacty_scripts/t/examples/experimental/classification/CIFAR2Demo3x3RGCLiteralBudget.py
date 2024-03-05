from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time
import ssl
from skimage.color import rgb2hsv

ssl._create_default_https_context = ssl._create_unverified_context

from keras.datasets import cifar10

animals = np.array([2, 3, 4, 5, 6, 7])

max_included_literals = 32
clauses = 8000
T = int(clauses * 0.75)
s = 10.0
patch_size = 3
resolution = 8
number_of_state_bits_ta = 8
literal_drop_p = 0.0

epochs = 250
ensembles = 5

classes = 10

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

X_train_org = X_train_org.astype(np.uint32)
X_test_org = X_test_org.astype(np.uint32)

X_train_org += 1
X_test_org += 1

X_train_r = 1.0 * X_train_org[:,:,:,0] / X_train_org.sum(axis=3)
X_train_g = 1.0 * X_train_org[:,:,:,1] / X_train_org.sum(axis=3)
X_train_sum =  X_train_org.sum(axis=3)

X_test_r = 1.0 * X_test_org[:,:,:,0] / X_test_org.sum(axis=3)
X_test_g = 1.0 * X_test_org[:,:,:,1] / X_test_org.sum(axis=3)
X_test_sum =  X_test_org.sum(axis=3)

Y_train = Y_train.reshape(Y_train.shape[0])
Y_test = Y_test.reshape(Y_test.shape[0])

X_train = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3, resolution),
                   dtype=np.uint32)
for z in range(resolution):
    X_train[:, :, :, 0, z] = X_train_r[:, :, :] >= (z + 1) / (resolution + 1)
    X_train[:, :, :, 1, z] = X_train_g[:, :, :] >= (z + 1) / (resolution + 1)
    X_train[:, :, :, 2, z] = X_train_sum[:, :, :] >= (z + 1)*256*3 / (resolution + 1)

X_test = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3, resolution),
                  dtype=np.uint32)

for z in range(resolution):
    X_test[:, :, :, 0, z] = X_test_r[:, :, :] >= (z + 1) / (resolution + 1)
    X_test[:, :, :, 1, z] = X_test_g[:, :, :] >= (z + 1) / (resolution + 1)
    X_test[:, :, :, 2, z] = X_test_sum[:, :, :] >= (z + 1)*256*3 / (resolution + 1)

X_train = X_train.reshape((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3 * resolution))
X_test = X_test.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3 * resolution))

Y_train = np.where(np.isin(Y_train, animals), 1, 0)
Y_test = np.where(np.isin(Y_test, animals), 1, 0)

f = open("cifar2_%.1f_%d_%d_%d_%.2f_%d_%d.txt" % (
s, clauses, T, patch_size, literal_drop_p, resolution, max_included_literals), "w+")
for ensemble in range(ensembles):
    tm = TMClassifier(clauses, T, s, platform='GPU', patch_dim=(patch_size, patch_size),
                      number_of_state_bits_ta=number_of_state_bits_ta, weighted_clauses=True,
                      literal_drop_p=literal_drop_p, max_included_literals=max_included_literals)
    for epoch in range(epochs):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        result_test = 100 * (tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        result_train = 100 * (tm.predict(X_train) == Y_train).mean()

        number_of_includes = 0
        for i in range(2):
            for j in range(clauses):
                number_of_includes += tm.number_of_include_actions(i, j)
        number_of_includes /= 2 * clauses

        print("%d %d %.2f %.2f %.2f %.2f %.2f" % (
        ensemble, epoch, number_of_includes, result_train, result_test, stop_training - start_training, stop_testing - start_testing))
        print("%d %d %.2f %.2f %.2f %.2f %.2f" % (
        ensemble, epoch, number_of_includes, result_train, result_test, stop_training - start_training, stop_testing - start_testing),
              file=f)
        f.flush()
f.close()
