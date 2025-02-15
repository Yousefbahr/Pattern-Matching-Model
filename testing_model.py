from utils import *
from model import Tx, char_to_index

loaded_model = load_model("model.keras", custom_objects={"Model":Model})

targets = pd.read_excel("Product Matching Dataset.xlsx", sheet_name="Master File")["product_name_ar"]

# to predict the similarity between a list of unformatted and target formatted names, uncomment the following
input = np.array( ["الفانوفا ركزززز",
        "بانادووووول"])

df, preds = get_prediction(loaded_model, input, Tx, char_to_index, targets)

print(df)

