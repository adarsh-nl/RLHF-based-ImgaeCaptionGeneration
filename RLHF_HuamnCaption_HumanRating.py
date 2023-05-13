import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.applications.xception import Xception, preprocess_input
from pickle import dump, load
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import torch
import torch.optim as optim

max_length = 32
dataset_images = "/content/drive/MyDrive/CV Project - Image caption generator/archive (1)/Images"
tokenizer = load(open("/content/drive/MyDrive/CV Project - Image caption generator/Models/tokenizer.p","rb"))
model = load_model('/content/drive/MyDrive/CV Project - Image caption generator/Models/model_11.h5')
xception_model = Xception(include_top=False, pooling="avg")

# Initialize RL model parameters
alpha = 0.1 # Learning rate
epsilon = 0.1 # Exploration rate
gamma = 0.9 # Discount factor
q_table = np.random.rand(max_length, len(tokenizer.word_index) + 1)
RL_learning_loop = 0

def custom_loss(cross_entropy, feedback):
  custom_loss = cross_entropy - feedback
  return custom_loss

def calculate_cross_entropy(string1, string2):
    # Create tokenizer and fit on the strings
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([string1, string2])

    # Convert strings to sequences of numerical indices
    sequences1 = tokenizer.texts_to_sequences([string1])[0]
    sequences2 = tokenizer.texts_to_sequences([string2])[0]

    # Pad sequences to the same length
    max_length = max(len(sequences1), len(sequences2))
    sequences1 = pad_sequences([sequences1], maxlen=max_length, padding='post')
    sequences2 = pad_sequences([sequences2], maxlen=max_length, padding='post')

    # Convert sequences to one-hot encoded vectors
    one_hot1 = tf.one_hot(sequences1, len(tokenizer.word_index) + 1)
    one_hot2 = tf.one_hot(sequences2, len(tokenizer.word_index) + 1)

    # Calculate categorical cross-entropy
    cross_entropy = tf.keras.losses.categorical_crossentropy(one_hot1, one_hot2)
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    
    return mean_cross_entropy.numpy()
  
# Update model parameters using RL algorithm
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
for img in os.listdir(dataset_images): # Number of RL iterations
    RL_learning_loop += 1
    if RL_learning_loop < 10:
      image_path = dataset_images + "/" + img
      image = Image.open(image_path)
      #image = image.resize(300,300)
      image.show()
      photo = extract_features(image_path, xception_model)
      in_text = 'start'
      for i in range(max_length):
          sequence = tokenizer.texts_to_sequences([in_text])[0]
          sequence = pad_sequences([sequence], maxlen=max_length)
          preds = model.predict([photo,sequence], verbose=0)
          next_index = sample_pred(preds, temperature=0.5) # introduce randomness with temperature=0.5
          next_word = word_for_id(next_index, tokenizer)
          if next_word is None:
              break
          in_text += ' ' + next_word
          if next_word == 'end':
              break

      model_caption = in_text
      print("Generated caption:", in_text)
      human_caption = input("Enter Human Caption")

      cross_entropy = calculate_cross_entropy(model_caption, human_caption)
      print("The cross_entropy betwee two stringa are:", cross_entropy)

      #Huma feedback 
      feedback = float(input("Rate the quality of the caption (-1 or +1): "))
      loss = custom_loss(cross_entropy, feedback)
      
      # Compute gradients
      variables = model.trainable_variables

      gradients = []
      # Get the "embedding_1" layer
      embedding_layer = model.get_layer('embedding_1')

      # Get the weights of the "embedding_1" layer
      weights = embedding_layer.get_weights()[0]
      weights = np.array(weights)
      print("Shape of the weights: {}".format(weights.shape))
      # Calculate the gradient using chain rule
      gradients = np.multiply(loss, np.ones_like(weights))
      # Reshape the gradients to match the shape of the weights
      gradients = gradients.reshape(weights.shape)
      
      # Update weights using optimizer
      weights -= optimizer.learning_rate * gradients
      print("Shape of the weights after optimisation: {}".format(weights.shape))
      embedding_layer.set_weights([weights])
      print("**--Weights updated--**")
      # Save updated model and Q-table
      model.save('models/model_rl.h5')

      # Save human captions to the file
      with open(human_captions_file, 'a') as file:
        file.write(human_caption + "\n")
