import os.path
import sys
import datetime
import time

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

import numpy as np
from pysnaike import activations, layers, models, callbacks, constraints

from matplotlib import pyplot as plt                    # Used for creating plots
from matplotlib import style                            # Graph style
style.use('fivethirtyeight')

num_images = 10000
acc = np.array([])

def load_mnist_data(file_path=None):
    '''
        Load mnist labeled handwritten digits and split into training data and test data.
    '''
    if file_path:
        print("existing file")
        data = np.load(file_path, allow_pickle=True)
        x_train = data['x_train'][:num_images]
        x_val = data['x_val'][:num_images]
        y_train = data['y_train'][:num_images]
        y_val = data['y_val'][:num_images]
    else:
        # To avoid import waiting time if not necessary
        from sklearn.datasets import fetch_openml               # Load mnist dataset
        from keras.utils.np_utils import to_categorical         # Convert to one-hot encoded labels
        from sklearn.model_selection import train_test_split    # Split dataset into training and test

        # Load data and labels from mnist dataset
        x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        # Normalize to values between 0 and 1
        x = (x/255).astype('float32')
        # Convert labels to one-hot encoded labels
        y = to_categorical(y)

        # Split dataset into training data and test data
        # [x_train, x_val, y_train, y_val]
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)
        num_examples = 10000
        x_train = np.array(x_train)[:num_examples]
        x_val = np.array(x_val)[:num_examples]
        y_train = np.array(y_train)[:num_examples]
        y_val = np.array(y_val)[:num_examples]
        # data = np.array([x_train, x_val, y_train, y_val])
        print("saving file")
        np.savez("10000_imgs.npz", x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val)
        print("saved")
    print(x_train.shape)
    return [x_train, x_val, y_train, y_val]

def train_example_callback(**kvargs):
    print(f"Train example {kvargs['num_in_batch']} in batch {kvargs['num_batch']}")

def epoch_callback(**kvargs):
    global acc
    ax = plt.figure(1).add_subplot(1,1,1)
    plt.figure(1).set_size_inches(6,3)
    print(f"Epoch {kvargs['num_epoch']}")
    print("Saving params ...")
    np.savez("mnist_model.npz", **M.params)
    accuracy = M.compute_accuracy(dataset[1][:200], dataset[3][:200])
    print(f"accuracy: {accuracy}")
    # acc.append(accuracy)
    acc = np.append(acc, accuracy)
    # Draw the training statistics to an interactive diagram
    
    # fig = plt.figure(1)
    # plt.cla()
    # ax.plot(range(0, acc.shape[0]), acc * 100)
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("% correkt")
    # fig.suptitle('Algorithm accuracy for each epoch', fontsize=16)
    # # for i in range(len(acc)):
    # #     ax.annotate('{} = {}'.format(i + 1, np.round(acc[i] * 100, 1)), (i, acc[i] * 100 + 0.4), ha='center', va='center')
    # k = acc.shape[0] - 1
    # ax.annotate('Epoch {} = {}%'.format(k + 1, np.round(acc[k] * 100, 1)), (k, acc[k] * 100), ha='center', va='center')
    # plt.draw()
    # plt.pause(.001)


print('Loading data...')
dataset = load_mnist_data(file_path="10000_imgs.npz")

M = models.Sequential()

# M.add(layers.Reshape((1,28,28), input_shape=(1,784)))

# M.add(layers.Conv2D(32, kernel_size=(3,3), input_shape=(1,28,28), activation=activations.leaky_relu, kernel_constraint=constraints.max_norm(4)))
# M.add(layers.MaxPooling2D((2,2)))
# M.add(layers.Dropout(0.5))
# M.add(layers.Conv2D(32, kernel_size=(3,3), input_shape=(32,14,14), activation=activations.leaky_relu, kernel_constraint=constraints.max_norm(4)))
# M.add(layers.MaxPooling2D((2,2)))
# M.add(layers.Dropout(0.5))
# M.add(layers.Flatten())
# M.add(layers.Dense(128, activation=activations.leaky_relu, kernel_constraint=constraints.max_norm(4)))
# M.add(layers.Dropout(0.8))
# M.add(layers.Dense(10, activation=activations.softmax, kernel_constraint=constraints.max_norm(4)))

# M.add(layers.Reshape((784), input_shape=(1,28,28)))

M.add(layers.Conv2D(25, kernel_size=(3,3), input_shape=(1,28,28), padding="valid", activation=activations.leaky_relu, kernel_constraint=constraints.max_norm(4)))
M.add(layers.MaxPooling2D((2,2)))
M.add(layers.Flatten())

M.add(layers.Dense(100, activation=activations.leaky_relu, kernel_constraint=constraints.max_norm(4)))
M.add(layers.Dense(10, activation=activations.softmax, kernel_constraint=constraints.max_norm(4)))

# M.add(layers.Conv2D(64, kernel_size=(3,3), input_shape=(32,14,14), activation=activations.leaky_relu, kernel_constraint=constraints.max_norm(4)))
# M.add(layers.Conv2D(64, kernel_size=(3,3), input_shape=(64,14,14), activation=activations.leaky_relu, kernel_constraint=constraints.max_norm(4)))
# M.add(layers.MaxPooling2D((2,2)))
# M.add(layers.Dropout(0.75))
# M.add(layers.Flatten())
# M.add(layers.Dense(256, activation=activations.leaky_relu, kernel_constraint=constraints.max_norm(4)))
# M.add(layers.Dense(84, activation=activations.leaky_relu, kernel_constraint=constraints.max_norm(4)))
# M.add(layers.Dropout(0.75))
# M.add(layers.Dense(10, activation=activations.softmax, kernel_constraint=constraints.max_norm(4)))

M.compile()
M.description()

params_path = "mnist_model.npz"
if os.path.exists(params_path):
    params = np.load(params_path, allow_pickle=True)
    for key in params.files:
        M.params[key] = params[key]

print("training ...")


my_callbacks = callbacks.Callbacks()
# my_callbacks.on("train_example", train_example_callback)
my_callbacks.on("epoch", epoch_callback)

print(dataset[0].shape)
dataset[0] = np.reshape(dataset[0], (num_images, 1, 28, 28))
dataset[1] = np.reshape(dataset[1], (num_images, 1, 28, 28))
# print("inputs:")
# inputs = dataset[0][2]
# print(inputs)
# print(inputs.shape)
# plt.imshow(np.sum(dataset[0][3], 0))
# plt.show()
# plt.imshow(np.sum(dataset[0][4], 0))
# plt.show()
# plt.imshow(np.sum(dataset[0][5], 0))
# plt.show()
# # print(np.sum(dataset[0][0]))
# print(dataset[2][3])
# print(dataset[2][4])
# print(dataset[2][5])
# Det er fordi jeg bruger sum i stedet for squeeze.... haha
# print(dataset[2][0])
# outputs = M.forward_pass(inputs)
# print("outputs:")
# print(outputs.shape)
# print(dataset[0].shape)
start_time = datetime.datetime.now()

M.train(dataset[0], dataset[2], epochs=100, mini_b_size=1, learning_rate=0.001, optimizer='BATCH', mini_b_shuffle=True, callbacks=my_callbacks)
end_time = datetime.datetime.now()
print(f"Time: {end_time - start_time}")

# print("done")
# accuracy = M.compute_accuracy(dataset[1][:num_images], dataset[3][:num_images])
# print(f"Final accuracy: {accuracy}")
# print(accuracy)
