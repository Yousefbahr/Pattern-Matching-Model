import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Softmax, Reshape, Embedding
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda, Masking, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from fuzzywuzzy import fuzz, process

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

def apply_regex(sentence):
  """
  apply regex on a string and return new one
  """
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
  return new


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
    x[i] = apply_regex(sentence)

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

def pre_process_dataset_training(X, Y, Tx, char_to_index, unique_classes = 500, sample_per_categ = 200 , negative_samples_ratio=0.5):
  """
  X: array of input strings (samples) of shape(m,) | where m is number of samples
  Y: array of target strings of shape (m,) | where m is number of samples
  Tx: sequence length
  char_to_index: a dict mapping all characters in vocab to their indices (0 to vocab_size)
  unique_classes: number of unqiue classes in the dataset
  sample_per_categ: is the number of positive and negative samples (combined) for each sample
  negative_samples_ratio: ratio of negative to postive samples

  returns 'sample_per_categ' positve and negative samples for each sample
  for negative samples randomly choose another class

  new_X: shape(n, 2, Tx)
  new_y: shape(n,)
  n is the new number of samples , which could be different that the original (m)
  n = sample_per_categ * unique_classes
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
  n = (sample_per_categ * unique_classes)
  new_X = np.zeros((n, 2, Tx))
  new_y = np.full((n,), -1)

  num_positive_samples = int(sample_per_categ * (1 - negative_samples_ratio))
  num_negative_samples =  int(sample_per_categ * negative_samples_ratio)

  # positive samples
  for i in range(unique_classes): # 0 to 499
    for j in range(num_positive_samples): # 0 to 239

      first_index = start_new_category[i]
      # last category
      if i == len(start_new_category) - 1:
        last_index = len(X)
      else:
        last_index = start_new_category[i + 1] - 1

      # index for indices_X and indices_y
      index = np.random.randint(first_index, last_index, size = 1)

      # positive sample
      new_X[int(j + (i * sample_per_categ * (1 - negative_samples_ratio))) , 0, :] = indices_X[index, :]
      new_X[int(j + (i * sample_per_categ * (1 - negative_samples_ratio))) , 1, :] = indices_y[index, :]
      new_y[int(j + (i * sample_per_categ * (1 - negative_samples_ratio)))] = 1


  # starting index
  start = unique_classes * (sample_per_categ * (1 - negative_samples_ratio))
  # negative samples
  for i in range(unique_classes): # 0 to 499
    for j in range(num_negative_samples): # 0 to 239

      first_index = start_new_category[i]
      # last category
      if i == len(start_new_category) - 1:
        last_index = len(X)
      else:
        last_index = start_new_category[i + 1] - 1

      # index for indices_X and indices_y
      # choose index from current category
      index = np.random.randint(first_index, last_index, size = 1)

      # random index for negative samples (not similar)
      rand_index = np.random.randint(low= 0, high= X.shape[0], size=1)[0]
      # if the random index is the same category as current category, get another index
      while first_index <= rand_index < last_index:
        rand_index = np.random.randint(low= 0, high= X.shape[0], size=1)[0]


      # negative sample
      new_X[int(j + start + (i * sample_per_categ * negative_samples_ratio)) , 0, :] = indices_X[index, :]
      new_X[int(j + start + (i * sample_per_categ * negative_samples_ratio)) , 1, :] = indices_y[rand_index, :]
      new_y[int(j + start + (i * sample_per_categ * negative_samples_ratio))] = 0

  return new_X[new_y != -1], new_y[new_y != -1]


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

def get_binary_preds(pred_sentences, input, pred_numbers):
  """
  pred_sentences: numpy array of top predicted sentences
  input: input_sentence -> string
  pred_numbers: the model's predictions of each name in pred_sentences
  """
  if isinstance(input, (pd.DataFrame, pd.Series)):
    input = input.values

  if isinstance(pred_sentences, (pd.DataFrame, pd.Series)):
    pred_sentences = pred_sentences.values

  assert pred_sentences.ndim == 1, "Not a numpy array"

  input = apply_regex(input)

  # add spaces between numbers to make ease matching
  pattern = r'(?<=[0-9])(?=[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF])|(?<=[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF])(?=[0-9])'
  input = re.sub(pattern, ' ', input)

  for i in range(len(pred_sentences.copy())):
    pred_sentences[i] = re.sub(pattern, ' ', pred_sentences[i])

  matches = process.extract(input, pred_sentences, scorer=fuzz.token_set_ratio, limit=5)

  fuzzy_pred = matches[0][1]

  idx = pred_sentences == matches[0][0]
  model_pred = pred_numbers[idx][0] * 100

  threshold = 55

  # if fuzzy similarity score less than model's similarity score , take the model's predictions
  if fuzzy_pred < threshold:
    return (pred_sentences[idx][0], model_pred) , np.array(model_pred > 95).astype(np.float32)

  return matches[0], np.array(float(matches[0][1]) > threshold).astype(np.float32)

def get_prediction(model, input_sentences, Tx, char_to_index , targets):
  """
  model: the loaded model
  input_sentences: array of strings of medication names
  Tx: sequence length
  char_to_index: a dict mapping characters to indices
  targets: 1D array of formatted arabic target names

  get prediction using 'input_sentences' from 'targets' array

  Returns:
    - Pandas DataFrame : contains input sentences , matched strings, and probability
     - Binary predictions : 1D numpy array of shape (len(input_sentences) * len(targets)) ,
                      this array only contains 0 or 1. every name in 'input_sentences' is
                      matched against every name in 'targets'. If similar then 1 , else 0
  """
  assert targets.ndim == 1 , "Targets array is not 1-dimensional"
  assert input_sentences.ndim == 1 , "Input array is not 1-dimensional"

  if isinstance(targets, (pd.DataFrame, pd.Series)):
    targets = targets.values

  if isinstance(input_sentences, (pd.DataFrame, pd.Series)):
    input_sentences = input_sentences.values

  # clean target names
  target_to_idx = {}
  pattern = r'(?<=[0-9])(?=[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF])|(?<=[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF])(?=[0-9])'
  for idx, name in enumerate(targets):
    name = re.sub(pattern, ' ', name)
    target_to_idx.update({name: idx})

  # shape (targets[0], Tx)
  targets_indices = np.expand_dims(pre_process_strings(targets, Tx, char_to_index), 1)

  input_indices = pre_process_strings(input_sentences, Tx, char_to_index)

  # repeat input to try to match it with each sentence in targets
  input_repeated = np.expand_dims(np.repeat(input_indices, targets.shape[0], axis=0), 1)

  targets_indices_repeated = np.tile(targets_indices, (len(input_sentences), 1, 1))

  concat_indices = np.concatenate((targets_indices_repeated, input_repeated), axis=1)

  model_input = tf.one_hot(concat_indices, depth =len(char_to_index), axis=-1)
  # model predictions
  preds = model.predict(model_input, batch_size=1024).reshape(len(input_sentences), targets.shape[0])

  # true 0/1 predictions
  binary_preds = np.zeros((len(input_sentences) * targets.shape[0],))

  # true indices for targets
  for_targets = []

  probabs = []

  # get indices of the top 5 predictions
  top_indices = np.argpartition(preds, -5)[:, -5:]

  for i in range(len(input_sentences)):
    matched , binary = get_binary_preds(targets[top_indices[i]] , input_sentences[i], preds[i, top_indices[i]] )

    idx = target_to_idx[matched[0]]

    for_targets.append(idx)

    binary_preds[ idx + (i * 1000)] = binary

    probabs.append(matched[1])


  return pd.DataFrame({
      "Probability": probabs,
      "Matched": targets[for_targets],
      "Input":input_sentences,
  }, index = range(len(input_sentences))) , binary_preds


