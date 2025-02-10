from utils import *
from model import Tx, char_to_index

loaded_model = load_model("model.keras", custom_objects={"Model":Model})

# to predict the similarity of two names , uncomment the following
"""
name1 = "بانادزل"
name2 = "بانادول"
input = pre_process_input(name1, name2, Tx, char_to_index)
pred = loaded_model.predict(input, batch_size=64)
probab = np.max(pred)
print(probab)
"""

# to predict the similarity between a list of unformatted and target formatted names, uncomment the following
"""

targets = np.array(["الفانوفا 20 قرص"
          , "بانادول اكسترا 30 قرص "])


input = np.array( ["الفانوفا ركزززز",
        "بانادووووول"])

get_prediction(loaded_model, input, Tx, char_to_index, targets)
"""
