
from utils import *

# the input sequence will be 30
Tx = 30

arabic_char = list("اﻷإﻹأآءئبتثجحخدذرزسشصضطظعغفقكلمنهويىة")
english_char = list("abcdefghijklmnopqrstuvwxyz")
symbols = list(".%/")
numbers = [str(num) for num in range(10)]
# arabic + symbols + english + numbers

# mapping characters to indices
index_to_char = {}
char_to_index = {}
for i, char in enumerate(arabic_char + symbols + english_char + numbers):
  index_to_char.update({i: char })
  char_to_index.update({char: i})

vocab_size = len(index_to_char)


def main():
    # sku 307 is not in the dataset file
    data = pd.read_excel("data.xlsx", sheet_name="Dataset")
    target = pd.read_excel("data.xlsx", sheet_name="Master File")


    seller_item_name = data["seller_item_name"]
    sku_values = data["sku"]


    unique_categ = len(sku_values.unique())
    X, y = pre_process_dataset_training(data["seller_item_name"], data["marketplace_product_name_ar"],
                           Tx, char_to_index, unique_categ)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    X_train = tf.one_hot(X_train, depth = vocab_size, axis = -1)
    X_test =  tf.one_hot(X_test, depth = vocab_size, axis = -1)

    """
    - The input data X : consists of a tuple (inp1, inp2)
            - Each input represents a sentence (medication name)
            
    -  The target y : is 0 or 1 for similar or dissimilar, a binary classification problem.
    
    - The architecture is (for each input, shared weights) a masking layer , LSTM layer, a distance function which computes the euclidean distance between the final hidden states of the two lstms then Dense Layer with sigmoid activation.
    """

    model = Model(1, 64)

    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs= 4, verbose=1, batch_size=64)

    preds = model.predict(X_test, batch_size=64)
    binary_preds = (preds > 0.5).astype(np.float32).squeeze()

    print(accuracy_score(y_test,binary_preds))

    sen = "الفانوفا"
    sen2 = "الفانوفا بلس"
    sen3 = "  بانادول اكسترا  48"
    sen4 = "اسبوسيد اطفال 75"
    sen5 = " مش موجود جديييد"

    get_prediction(model, np.array([sen,  sen2, sen3, sen4, sen5]), Tx, char_to_index, "data.xlsx")

if __name__ == "__main__":
    main()
