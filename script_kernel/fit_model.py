import pickle

import numpy as np
from grakel.kernels import GraphletSampling
from sklearn.svm import SVC

# load data
data = pickle.load(open('data/kernel/data.pkl', 'rb'))
G_train = data['G_train']
G_test = data['G_test']
y_train = data['y_train']
y_test = data['y_test']

# fit the kernel
kernel = GraphletSampling(k=5)
K_train = kernel.fit_transform(G_train)
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)

# evaluate on the test set
K_test = kernel.transform(G_test)
y_pred = clf.predict(K_test)
test_accuracy = np.mean(y_pred == y_test)
print(f'Test set accuracy {test_accuracy}')
