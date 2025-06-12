import imageio

def plot_activation_map(activation_map, i, label, example_index):
  # Create heatmap from the data
  hm = sns.heatmap(data=activation_map, square=True ,cmap='Greens', cbar=False) #, yticklabels=y_labels, )

  # Add labels to the x and y axes
  plt.xlabel('x-axis')
  plt.ylabel('y-axis')
  plt.title(f'Activation Map for Layer {i}')

  # Show the plot
  #plt.show()

  # Convert plot to np array
  fig = plt.gcf()
  fig.canvas.draw()
  img_array = np.array(fig.canvas.renderer.buffer_rgba())
  plt.close(fig)

  # Save the frame as a PNG image
  filename = f"example_{label}_{example_index}_frame_{i}.png"
  imageio.imwrite(filename, img_array)

  return filename

def generate_gif(filenames, gif_filename):
  # combine images into a GIF
  images = []
  for filename in filenames:
    images.append(imageio.imread(filename))
    imageio.mimsave(gif_filename, images, fps=1, loop=0) # Adjust fps as needed

def generate_images(activations, label, example_index):
  filenames = []

  for i in range(0, len(activations)-1):
    #print(activations[i].get_shape())
    #print(np.average(activations[i][0]))
    activation_map = np.array(activations[i][example_index])
    #print(activation_map)
    activation_map = activation_map.reshape((28,28))
    filename = plot_activation_map(activation_map, i, label, example_index)
    filenames.append(filename)

  return filenames

gifs = {}

for label in range(len(label_names)):
  indices = dic_examples[label]
  print(f'Indices for Examples of digit{label}:', indices)
  gifs[label] = []
  for example_index in indices:
    filenames = generate_images(activations, label, example_index)
    gif_filename = f'gifs/example_{label}_{example_index}.gif'
    generate_gif(filenames, gif_filename)
    gifs[label].append(gif_filename)

#example_index = 20
#filenames = generate_images(activations, example_index)
#generate_gif(filenames, 'animation.gif')

'''
# combine images into a GIF
images = []
for filename in filenames:
  images.append(imageio.imread(filename))
  imageio.mimsave('animation.gif', images, fps=1) # Adjust fps as needed
'''

# clean up
#for filename in filenames:
#  os.remove(filename)

'''
# get_weights()[0] contains the weights, get_weights()[1] contains the biases
weights = np.array(model.layers[layer_num].get_weights()[0])

weights_for_node = weights[:,node_num]
weights_for_node = weights_for_node.reshape((28,28))  # reshape to 28x28 pixel image
#print(weights_for_node)
'''




'''
' Display Stop Motion Images
'''
from IPython import display
from IPython.display import HTML

def display_examples(gif_list):
  for gif in gif_list:
    display.display(display.Image(gif))

print(gifs[0])
display_examples(gifs[0])


def display_image(path):
  return f'<img src="/content/{path}" width="100" height="100" >'
  #return f'<img src="https://fastly.picsum.photos/id/237/200/300.jpg?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U" width="100" height="100" >'

df = pd.DataFrame({'Image': gifs[0]})
df['Image'] = df['Image'].apply(display_image)
html = df.to_html(escape=False)
print(html)

display.HTML(html)