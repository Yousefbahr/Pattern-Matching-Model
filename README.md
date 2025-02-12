# Product-Matching-Model
A pattern-matching model for identifying medication names, utilizing a Siamese network. This model takes an input, which can be in either Arabic or English, and computes a similarity score between the input and a database of formatted medication names. The Siamese network compares the input to the database entries and determines the degree of similarity, helping to identify the most relevant matches.

**The model achieved an accuracy of 99% on test data.**

**The process of finding a match takes between 200 and 400 milliseconds.**


## Architecture
The model is designed to process two input sequences. The network includes the following components:

- Input Layer: Two input sequences are passed into the model.
- Masking Layer: Each input sequence is processed through a masking layer to handle variable-length sequences, which masks the padding tokens.
- LSTM Layer: The masked inputs are fed into a Long Short-Term Memory (LSTM) layer to extract relevant features.
- Euclidean Distance Layer: The hidden states of the LSTM for both input sequences are compared using the Euclidean distance function to measure their similarity.
- Dense Layer: The result of the Euclidean distance is passed through a Dense layer with a sigmoid activation function, providing a final similarity score between the two input sequences.

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
    
### Predicting the similarity between two names
In the `testing_model.py` file, uncomment the following and change the names of the two variables `name1` and `name2` to whichever you want and run the file.
````
    loaded_model = load_model("model.keras", custom_objects={"Model":Model})
    name1 = "بانادزل"
    name2 = "بانادول"
    input = pre_process_input(name1, name2, Tx, char_to_index)
    pred = loaded_model.predict(input)
    probab = np.max(pred)
  ````

### Predicting the similarity between a list of unformatted and target formatted names

In the `testing_model` file, uncomment the following and provide a numpy 1D array of names for the input and another numpy 1D array of targets.
- `get_prediction` function takes a model instance, a numpy array of names for the inputs and another of targets.
 It outputs a `Panda DataFrame` of the inputs, matched names, and the probability for each input name.
- You could provide a list of input names and/or targets from a file
- Both arrays must be numpy.
  
Here's a demo.
````
targets = np.array(["الفانوفا 20 قرص"
          , "بانادول اكسترا 30 قرص "])


input = np.array( ["الفانوفا ركزززز",
        "بانادووووول"])

get_prediction(loaded_model, input, Tx, char_to_index, targets)
````
Output:

| Index       | Probability | Matched   |  Input   |
|------------|--------------|--------------| --------------|
| 0      | 0.986129  |     الفانوفا 20 قرص  |الفانوفا ركزززز   |
| 1        | 0.984625  |  بانادول اكسترا 30 قرص   | بانادووووول	     |

                                              




 
