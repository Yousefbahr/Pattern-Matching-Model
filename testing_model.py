
from utils import *
from model import Tx, char_to_index

loaded_model = load_model("model.keras", custom_objects={"Model":Model})

target_file = "Product Matching Dataset.xlsx"
input_file = "input_file.xlsx"

# to predict the similarity between a list of unformatted and target formatted names, uncomment the following
df, preds = get_prediction(loaded_model, input_file, Tx, char_to_index, target_file)

print(df)

