# Product-Matching-Model
The model uses a Siamese network for pattern matching to identify medication names. It accepts inputs in either Arabic or English and calculates a similarity score between the input and entries in a database of formatted medication names. The Siamese network compares the input against each database entry and computes a similarity degree, returning the top k most similar candidates. To refine this process, a string matching algorithm is applied to identify and return the most similar medication name.

- **When the model identifies a match as similar, it is accurate 96% of the time.**
- **The model successfully classifies 96% of all similar matches correctly.**
- **The process of finding a match takes between 200 and 400 milliseconds.**


## Architecture
The model is designed to process two input sequences. The network includes the following components:

- Input Layer: Two input sequences are passed into the model.
- Masking Layer: Each input sequence is processed through a masking layer to handle variable-length sequences, which masks the padding tokens.
- LSTM Layer: The masked inputs are fed into a Long Short-Term Memory (LSTM) layer to extract relevant features.
- Euclidean Distance Layer: The hidden states of the LSTM for both input sequences are compared using the Euclidean distance function to measure their similarity.
- Dense Layer: The result of the Euclidean distance is passed through a Dense layer with a sigmoid activation function, providing a final similarity score between the two input sequences.

The model returns the top k candidates, and a string-matching algorithm optimizes the selection by choosing the most accurate candidate from them.

## Input Layer
The input consists of both positive and negative samples processed from the dataset. 

The input shape for training is (m, 2, Tx, vocab_size) where: m is the batch size, Tx is the sequence length, vocab_size represents the size of the vocabulary and it is the length of the one-hot encoded representation used for the characters in a given name.

The target is of shape (m, ) where each value is either 1 or 0.

  
#### Positive Sample
  - A positive sample is where both input names are similar. One name is formatted, and the other is unformatted, but they are essentially the same.
  - The target for a positive sample is 1.

#### Negative Sample
  - A negative sample is where the input names are dissimilar. One name is formatted, and the other is unformatted, representing two different names.
  - The target for a negative sample is 0.
    
## Usage
    $ pip install -r requirements.txt

### Predicting the similarity between a list of unformatted and target formatted names

In the `testing_model.py` file, uncomment the following and provide an Excel file **with the same column names as in the file `input_file.xlsx`**, which represents the input names, and the Excel master file.

  - `input_file.xlsx` has 2 columns:
    - item_code: which isn't necessary for pattern matching
    - product_name: which consists of input unformatted names,

  - It outputs an Excel file with the same structure as  `input_file.xlsx` with two columns added:
    - sku: representing the code of the formatted name
    - item_name: formatted name
    
  - It outputs a `Panda DataFrame` of the inputs, matched names, and the probability for each input name.
  - It also outputs a prediction 1D array of length `number of input names * number of target names`. The input is matched with every name in the               'targets' array. The output is a binary array of only zeros or ones indicating similar or not.
  
- For more details, take a look at the Jupiter Notebook file `model.ipynb`.

Here's a demo.

````
from utils import *
from model import Tx, char_to_index

loaded_model = load_model("model.keras", custom_objects={"Model":Model})

target_file = "Product Matching Dataset.xlsx"
input_file = "input_file.xlsx"

# to predict the similarity between a list of unformatted and target formatted names, uncomment the following
df, preds = get_prediction(loaded_model, input_file, Tx, char_to_index, target_file)

 
