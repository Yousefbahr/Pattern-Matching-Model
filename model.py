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
    data = pd.read_excel("Product Matching Dataset.xlsx", sheet_name="Dataset")
    target = pd.read_excel("Product Matching Dataset.xlsx", sheet_name="Master File")


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

    get_prediction(model, np.array([sen,  sen2, sen3, sen4, sen5]), Tx, char_to_index, target["product_name_ar"])

    loaded_model = load_model("model.keras", custom_objects={"Model": Model})
    # Testing the model
    number_of_categ = 300

    start_new_categ = [0, 179, 358, 536, 714, 890, 1066, 1242, 1418, 1594, 1770, 1946, 2122, 2298, 2473, 2648, 2823,
                       2998, 3173, 3348, 3523, 3698, 3873, 4048, 4223, 4397, 4571, 4745, 4919, 5093, 5267, 5441, 5615,
                       5789, 5963, 6137, 6311, 6485, 6659, 6833, 7007, 7181, 7355, 7529, 7702, 7875, 8048, 8221, 8394,
                       8567, 8740, 8913, 9086, 9259, 9432, 9605, 9778, 9951, 10124, 10297, 10470, 10643, 10816, 10988,
                       11160, 11332, 11504, 11676, 11848, 12020, 12192, 12364, 12536, 12708, 12880, 13052, 13224, 13396,
                       13568, 13740, 13912, 14083, 14254, 14425, 14596, 14767, 14938, 15109, 15280, 15451, 15622, 15793,
                       15964, 16135, 16306, 16477, 16648, 16819, 16990, 17161, 17332, 17503, 17674, 17845, 18016, 18187,
                       18358, 18529, 18700, 18871, 19042, 19213, 19383, 19553, 19723, 19893, 20063, 20233, 20403, 20573,
                       20743, 20913, 21083, 21253, 21423, 21593, 21763, 21933, 22103, 22273, 22443, 22613, 22783, 22953,
                       23123, 23293, 23463, 23633, 23803, 23973, 24143, 24313, 24483, 24653, 24823, 24993, 25163, 25333,
                       25503, 25672, 25841, 26010, 26179, 26348, 26517, 26686, 26855, 27024, 27193, 27362, 27531, 27700,
                       27869, 28038, 28207, 28376, 28545, 28714, 28883, 29052, 29221, 29390, 29559, 29728, 29897, 30066,
                       30235, 30404, 30573, 30742, 30911, 31080, 31249, 31418, 31587, 31756, 31925, 32094, 32263, 32431,
                       32599, 32767, 32935, 33103, 33271, 33439, 33607, 33775, 33943, 34111, 34279, 34447, 34615, 34783,
                       34951, 35119, 35287, 35455, 35623, 35791, 35959, 36127, 36295, 36463, 36631, 36799, 36967, 37135,
                       37303, 37471, 37639, 37807, 37975, 38143, 38311, 38479, 38646, 38813, 38980, 39147, 39314, 39481,
                       39648, 39815, 39982, 40149, 40316, 40483, 40650, 40817, 40984, 41151, 41318, 41485, 41652, 41819,
                       41986, 42153, 42320, 42487, 42654, 42821, 42988, 43155, 43322, 43489, 43656, 43823, 43990, 44157,
                       44324, 44491, 44658, 44825, 44992, 45159, 45326, 45493, 45659, 45825, 45991, 46157, 46323, 46489,
                       46655, 46821, 46987, 47153, 47319, 47485, 47651, 47817, 47983, 48149, 48315, 48481, 48647, 48813,
                       48979, 49145, 49311, 49477, 49643, 49809, 49975, 50141, 50307, 50473, 50639, 50805, 50971, 51137,
                       51303, 51469, 51634, 51799, 51964, 52129, 52294, 52459, 52624, 52789, 52954, 53119, 53284, 53449,
                       53614, 53779, 53944, 54109, 54274, 54439, 54604, 54769, 54934, 55099, 55264, 55429, 55594, 55759,
                       55924, 56089, 56254, 56419, 56584, 56749, 56914, 57079, 57244, 57409, 57574, 57739, 57903, 58067,
                       58231, 58395, 58559, 58723, 58887, 59051, 59215, 59379, 59543, 59707, 59871, 60035, 60199, 60363,
                       60527, 60691, 60855, 61019, 61183, 61347, 61511, 61675, 61839, 62003, 62167, 62331, 62495, 62659,
                       62823, 62987, 63151, 63315, 63479, 63643, 63807, 63971, 64135, 64299, 64463, 64626, 64789, 64952,
                       65115, 65278, 65441, 65604, 65767, 65930, 66093, 66256, 66419, 66582, 66745, 66908, 67071, 67234,
                       67397, 67560, 67723, 67886, 68049, 68212, 68375, 68538, 68701, 68864, 69027, 69190, 69353, 69516,
                       69679, 69842, 70005, 70168, 70331, 70494, 70657, 70819, 70981, 71143, 71305, 71467, 71629, 71791,
                       71953, 72115, 72277, 72439, 72601, 72763, 72925, 73087, 73249, 73411, 73573, 73735, 73897, 74059,
                       74221, 74383, 74545, 74707, 74869, 75031, 75193, 75355, 75517, 75679, 75840, 76001, 76162, 76323,
                       76484, 76645, 76806, 76967, 77128, 77289, 77450, 77611, 77772, 77933, 78094, 78255, 78416, 78577,
                       78738, 78899, 79060, 79221, 79382, 79543, 79704, 79865, 80026, 80187, 80348, 80509, 80670, 80831,
                       80992, 81153, 81314, 81475, 81636, 81797, 81958, 82119, 82280, 82441, 82602, 82762, 82922, 83082,
                       83242, 83402]
    indices = []
    # random data points
    for i in range(number_of_categ):
        categ = np.random.randint(0, 499)
        indices.append(np.random.randint(start_new_categ[categ], start_new_categ[categ + 1] - 1))

    x_test_best = seller_item_name[indices]

    df, preds = get_prediction(loaded_model,
                               x_test_best,
                               Tx,
                               char_to_index,
                               target["product_name_ar"])

    y_test_best = np.zeros((len(x_test_best) * 1000,))
    # get true label test set
    for i in range(len(x_test_best)):
        sku = sku_values[indices[i]]
        index_master_file = np.where(target["sku"] == sku)[0][0] + (i * 1000)
        y_test_best[index_master_file] = 1

    print(df)
    print(confusion_matrix(y_test_best, preds))
    print(precision_score(y_test_best, preds))
    print(recall_score(y_test_best, preds))

if __name__ == "__main__":
    main()