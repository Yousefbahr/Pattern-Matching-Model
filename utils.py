import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Softmax, Reshape, Embedding
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda, Masking, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

class Model(tf.keras.Model):
  def __init__(self, num_of_layers, hidden_dim ,dropout_rate = 0.3, **kwargs):
    super(Model, self).__init__(**kwargs)
    self.num_layers = num_of_layers
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout_rate
    self.masking = Masking(mask_value = 0)
    self.last_lstm = LSTM(hidden_dim, return_state = True)
    self.lstm_layers = [LSTM(hidden_dim, return_sequences=True)
                        for _ in range(self.num_layers - 1)]
    self.dropout = Dropout(dropout_rate)
    self.dense = Dense(1, activation="sigmoid")

  def call_once(self, x):
    x = self.masking(x)

    for i in range(len(self.lstm_layers)):
      x = self.lstm_layers[i](x)
      x = self.dropout(x)

    x, final_hidden, _ = self.last_lstm(x)

    return final_hidden

  def call(self, inputs):
    x1 = inputs[:, 0, :, :]
    x2 = inputs[:, 1, :, :]

    first_output = self.call_once(x1)
    sec_output = self.call_once(x2)

    distance = tf.norm(first_output - sec_output, axis =1 , keepdims=True)
    distance = tf.clip_by_value(distance, clip_value_min=1e-7, clip_value_max=1e7)

    return self.dense(distance)

  def get_config(self):
    # Return the configuration of the custom layer, including any parameters
    config = super(Model, self).get_config()
    # Add any custom parameters here
    config.update({
        "num_of_layers":self.num_layers,
        "hidden_dim": self.hidden_dim,
        "dropout_rate" : self.dropout_rate
    })
    return config


def pre_process_strings(x, Tx, char_to_index, pad_value= -1):
  """
  x: list of strings
  Tx: sequence length
  char_to_index: a dict mapping characters to indices
  pad_value : the value to use in padding

  Processes a list of strings by applying regex then mapping the chars to indices

  Returns:
    indices: shape(x.shape[0], Tx)
  """

  for i, sentence in enumerate(x.copy()):
    # remove unwanted symbols
    new = re.sub(r"[^a-zA-Z0-9%\.\u0600-\u06FF\s]", " ", sentence)
    # remove tatweel symbol
    new = re.sub(r"\u0640", "" , new)
    # remove tashkeel
    new = re.sub(r"[\u064B-\u0652]", "", new)
    # remove arabic punctuation
    new = re.sub(r"[،؛؟“”‘’•]", "", new)
    # remove extra spaces
    new = re.sub(r"^\s+|\s+$|\s{2,}", " ", new).strip()
    new = re.sub(r"^\s+", " ", new)
    x[i] = new


  indices = np.full((len(x), Tx), pad_value)
  for i, sentence in enumerate(x):
    chars = list(sentence)
    for j, char in enumerate(chars):
      char = char.lower()
      # remove whitespace
      if char == ' ':
        continue
      if j < Tx:
        if char in char_to_index:
          indices[i, j] = char_to_index[char]


  return indices


def pre_process_dataset_training(X, Y,Tx, char_to_index, unique_classes = 500, num_pos_neg = 130):
  """
  X: array of input strings (samples) of shape(m,) | where m is number of samples
  Y: array of target strings of shape (m,) | where m is number of samples
  sku: array of sku of shape (m,)
  Tx: sequence length
  num_pos_neg: is the number of positive and negative samples for each sample
  char_to_index: a dict mapping all characters in vocab to their indices (0 to vocab_size)
  unique_classes: number of unqiue classes in the dataset

  returns 80 positve and negative samples for each sample
  for negative samples randomly choose another class

  new_X: shape(n, 2, Tx)
  new_y: shape(n,)
  n is the new number of samples , which could be different that the original (m)
  n = num_pos_neg * unique_classes
  """

  X = X.copy()
  Y = Y.copy()

  # -1 for padded values
  indices_X = pre_process_strings(X, Tx, char_to_index)
  indices_y = pre_process_strings(Y, Tx, char_to_index)

  start_new_category = []
  for i, sentence in enumerate(Y):
    # maintain the starting index of a new category
    if i == 0:
      start_new_category.append(i)
    else:
      if Y[i] != Y[i - 1]:
        start_new_category.append(i)


  # make positive and negative samples
  n = (num_pos_neg * 2 * unique_classes)
  new_X = np.zeros((n, 2, Tx))
  new_y = np.full((n,), -1)
  j = 0
  i = 0
  count = 0
  while(True):

    if count == n:
      break

    if count % (num_pos_neg * 2) == 0 and count != 0:
      j += 1
      if j == len(start_new_category) - 1:
        break
      i = start_new_category[j]

    new_X[count, 0, :] = indices_X[i, :]
    new_X[count, 1, :] = indices_y[i, :]
    new_y[count] = 1

    # random index for negative samples (not similar)
    rand_index = np.random.randint(low= 0, high= X.shape[0], size=1)[0]
    # if the random index is the same category as current category, get another index
    while np.array_equal(indices_y[rand_index, :], indices_y[i, :]):
      rand_index = np.random.randint(low= 0, high= X.shape[0], size=1)[0]

    new_X[count + 1, 0, :] = indices_X[i + 1, :]
    new_X[count + 1, 1, :] = indices_y[rand_index, :]
    new_y[count + 1] = 0

    i += 1
    count += 2

  return new_X, new_y


def pre_process_input(sentence1, sentence2, Tx, char_to_index):
  """
  sentence1: string
  sentence2: string
  Tx: sequence length
  char_to_index: a dict mapping all characters in vocab to their indices (0 to vocab_size)

  Returns:
    x: shape(1, 2, Tx, len(char_to_index)) , an input ready for prediction
  """

  sentences = [sentence1, sentence2]
  x = pre_process_strings(sentences, Tx, char_to_index)[np.newaxis, ...]

  return tf.one_hot(x, depth=len(char_to_index), axis=-1)


def get_prediction(model, input_sentences, Tx, char_to_index ,targets):
  """
  model: the loaded model
  input_sentences: array of strings of medication names
  Tx: sequence length
  char_to_index: a dict mapping characters to indices
  targets: 1D array of formatted arabic target names 
  
  get prediction using 'input_sentences' from the master file

  Returns:
    matched name, probability
  """

  assert targets.ndim == 1 , "Targets array is not 1-dimensional"
  assert input_sentences.ndim == 1 , "Input array is not 1-dimensional"
  
  # shape (targets[0], Tx)
  targets_indices = np.expand_dims(pre_process_strings(targets, Tx, char_to_index), 1)

  input_indices = pre_process_strings(input_sentences, Tx, char_to_index)

  # repeat input to try to match it with each sentence in targets
  input_repeated = np.expand_dims(np.repeat(input_indices, targets.shape[0], axis=0), 1)

  # targets_indices_repeated = np.concatenate([targets_indices, targets_indices], axis=0)

  targets_indices_repeated = np.tile(targets_indices, (len(input_sentences), 1, 1))

  concat_indices = np.concatenate((targets_indices_repeated, input_repeated), axis=1)

  model_input = tf.one_hot(concat_indices, depth =len(char_to_index), axis=-1)

  preds = model.predict(model_input, batch_size=64).reshape(len(input_sentences), targets.shape[0])

  # index of the biggest probab
  index = np.argmax(preds, axis = 1)

  return pd.DataFrame({
      "Probability": np.max(preds, 1),
      "Matched": targets[index],
      "Input":input_sentences,

  })
  # return np.max(preds, 1), targets[index]

