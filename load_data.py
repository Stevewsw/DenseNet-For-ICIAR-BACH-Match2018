# _*_coding: utf-8_*_

# PS C:\Users\aixin\Desktop\DenseNet> python .\run_cifar10.py
# Using TensorFlow backend.
# Network configuration:
# batch_size 64
# depth 7 
# dropout_rate 0.2
# growth_rate 12
# learning_rate 0.001
# nb_dense_block 1
# nb_epoch 30
# nb_filter 16
# plot_architecture False
# weight_decay 0.0001
# ./figures
# X_train-------> (50000, 32, 32, 3)
# X_train-------> (50000, 1)
# y_train[0]-------> [6]
# y_train[0].type-------> <class 'numpy.ndarray'>
# X_test-------> (10000, 32, 32, 3)
# y_test-------> (10000, 1)
# (50000, 32, 32, 3)


# X_train-------> (50000, 32, 32, 3)
# y_train-------> (50000, 1)
# y_train[0]-------> [6]
# y_train[0].type-------> <class 'numpy.ndarray'>
# y_train[0].dtype-------> uint8
# X_test-------> (10000, 32, 32, 3)
# y_test-------> (10000, 1)
# (50000, 32, 32, 3)

from __future__ import print_function

import numpy as np
train_file_name = r'C:\Users\aixin\Desktop\all_my_learning\match\npy_file\train.npy'
label_file_name = r'C:\Users\aixin\Desktop\all_my_learning\match\npy_file\label.npy'
def load_data(train_file_name, label_file_name):

    XT = np.load(train_file_name)
    YT = np.load(label_file_name)

    xt = np.load(train_file_name)
    yt = np.load(label_file_name)
    
    X_train = XT[:218//6*5000]
    y_train = YT[:218//6*5000]

    X_test = xt[218//6*5000:]
    y_test = yt[218//6*5000:]
    
    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data(train_file_name, label_file_name)




# X_train = XT[:218//6*5000]
# y_train = YT[:218//6*5000]

# X_test = xt[218//6*5000:]
# y_test = yt[218//6*5000:]

# print("XT.shape---", XT.shape)
# print("YT.shape---", YT.shape)
# print("xt.shape---", xt.shape)
# print("yt.shape---", yt.shape)
print(3*"=\n")

print("X_train.shape---", X_train.shape)
print("y_train.shape---", y_train.shape)
print("X_test.shape---", X_test.shape)
print("y_test.shape---", y_test.shape)
print(3*"=\n")


print("X_train.type---", type(X_train))
print("y_train.type---", type(y_train))
print("X_test.type---", type(X_test))
print("y_test.type---", type(y_test))
print(3*"=\n")

print("X_train[0]---", X_train[0])
print("X_test[0]---", X_test[0])
print("y_test[0]---", y_test[0])

print("X_test[0].dtype---", X_test[0].dtype)
print("y_test[0].dtype---", y_test[0].dtype)



# print("X_train------->", np.shape(X_train)
# print("X_train.type------>", type(X_train))
# print("X_train[0]------->",X_train[0])
# print("X_train[0].type------->",type(X_train[0]))
# print("X_train[0].dtype------->",X_train.dtype)



# print("y_train------->",np.shape(y_train)

# print("y_train.type------->",type(y_train))

# print("y_train[0]------->",y_train[0])

# print("y_train[0].type------->",type(y_train[0]))

# print("y_train[0].dtype------->",y_train.dtype)


# print(50*"=")

# print("X_test------->",np.shape(X_test)

# print("X_test.type------->",type(X_test))

# print("X_test[0]------->",X_test[0])

# print("X_test[0].type------->",type(X_test[0]))

# print("X_test[0].dtype------->",X_test.dtype)


# print(50*"=")

# print("y_test------->",np.shape(y_test)

# print("y_test.type------->",type(y_test))

# print("y_test[0]------->",y_test[0])

# print("y_test[0].type------->",type(y_test[0]))

# print("y_test[0].dtype------->",y_test.dtype)