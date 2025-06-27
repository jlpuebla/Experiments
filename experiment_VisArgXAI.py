import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import keras
import os
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder #not using
from sklearn import metrics
import collections
from utils.weight_utils import write_weights, generate_heatmap_for_weights_of_node


# load the dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print('size of training set=', Y_train.size)
print('size of test set=', Y_test.size)

# Flatten Y_train and Y_test as vector labels
Y_train_copy = Y_train.flatten()
Y_test_copy = Y_test.flatten()

# list to map label to number
label_names = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four',
               5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'}


def get_examples_of_label(y, number_of_examples, label_names, label):
  examples=[]
  for i in range(len(y)):
    if label_names[y[i]] == label:
      examples.append(i)
    if len(examples) >= number_of_examples:
      break
  return examples

def plot_images(dic, X, Y):
  # Plot images of the examples
  fig, axs = plt.subplots(nrows=10, ncols=5, figsize=(10,25))
  for key in dic:
    examples = dic[key]
    for i in range(len(examples)):
      image = X[examples[i]]
      label = Y[examples[i]]
      label_name = label_names[key]
      axs[key][i].imshow(image, cmap='gray')  # imshow renders a 2D grid
      axs[key][i].set_title(label_name)
      axs[key][i].axis('off')
  plt.show()

# get examples for each label
dic_examples={}
print(dic_examples)
for i in range(len(label_names)):
  dic_examples[i] = get_examples_of_label(Y_test_copy, 5, label_names, label_names[i])
print(dic_examples)
#plot_images(dic_examples, X_test, Y_test)

# Normalize dataset
X_train  = X_train/255
X_test  = X_test/255

# models
case1_nodes = [784,784,784,784,10]
case1_activ = ['relu','relu','relu','relu','softmax']


'''
' Create the model
'''
def create_model(input_shape, learning_rate, nodes, activations):
  tf.keras.backend.clear_session()
  tf.random.set_seed(0)

  model = Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=input_shape))

  # hidden layers
  for i in range(len(nodes)):
    model.add(Dense(units=nodes[i], activation=activations[i], use_bias=False))

  # compile model
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

  return model

# network parameters
number_of_layers = 8
number_of_nodes = 16
case_nodes = [number_of_nodes]*number_of_layers
case_nodes.append(10)
case_funcs = ['relu']*number_of_layers
case_funcs.append('softmax')
epochs = 200

# load existing model
path = 'models/model_8x16'
os.makedirs(path, exist_ok=True)

model_path = path +'/model_8x16.keras'
weights_path = path+'/model_8x16.weights.h5'
print(model_path)
print(weights_path)

if os.path.exists(model_path):
  print('Loading saved model.')
  # load saved model
  model = keras.models.load_model(model_path)
else:
  print('Create model and save it.')
  # Feed-Forward NN
  model = create_model(X_train[0].shape, 0.01, case_nodes, case_funcs)
  # train model
  model.fit(X_train, Y_train, epochs=epochs, verbose=2, batch_size=64, validation_split=0.2)
  # save model
  model.save(model_path)
  model.save_weights(weights_path)

model.summary()


def print_metrics(y_test, y_pred):
  # accuracy
  accuracy = metrics.accuracy_score(y_test, y_pred)
  print(f"accuracy={accuracy}")

  # confusion matrix
  confusion_matrix = tf.math.confusion_matrix(y_test, y_pred)
  print(confusion_matrix)

# prediction
Y_pred = np.argmax(model.predict(X_test), axis=-1)
print_metrics(Y_test, Y_pred)

counter = collections.Counter(Y_pred)
print('predicted: ', counter)
counter = collections.Counter(Y_test)
print('actual: ', counter)

write_weights(path, model)

# Output of each layer
print('Getting node activation values')
extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
activations = extractor(X_test)
#print(activations)

l = 0
print('Activation shapes')
for activation in activations:
  print(f'layer {l}: {activation.get_shape()}')
  l += 1

# Generate heatmaps for nodes 0, 6, 14 (which are the nodes activated in the first hidden layer)
for node in range(16):
  generate_heatmap_for_weights_of_node(model, 1, node)

