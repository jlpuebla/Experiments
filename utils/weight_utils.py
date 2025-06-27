import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

# write weights of model to text file
def write_weights(path, model):
  with open(path + "/weights.txt", "w") as f:  # "w" mode overwrites file
    for num, layer in enumerate(model.layers):
      print(f"Layer {num}")
      weights = layer.get_weights()
      
      if weights:
        f.write(f'Layer {num}:\n')
        f.write('weights:\n')
        f.write(np.array2string(weights[0], max_line_width=10000, separator=',', threshold=10000) + '\n')
        
        if len(weights) > 1:
          f.write('bias:\n')
          f.write(np.array2string(weights[1]) + '\n')

      else:
        print('no weights.')

# Visualize weights of first hidden layer 
def generate_heatmap_for_weights_of_node(model, layer_num, node_num):
  # get_weights()[0] contains the weights, get_weights()[1] contains the biases
  weights = np.array(model.layers[layer_num].get_weights()[0])

  weights_for_node = weights[:,node_num]
  weights_for_node = weights_for_node.reshape((28,28))  # reshape to 28x28 pixel image
  #print(weights_for_node)

  # Create heatmap from the data
  hm = sns.heatmap(data=weights_for_node, square=True ,cmap='Greens') #, yticklabels=y_labels, cbar=False)

  # Add labels to the x and y axes
  plt.xlabel('x-axis')
  plt.ylabel('y-axis')
  plt.title(f'Weight Map for Node {node_num}')

  # Show the plot
  plt.show()