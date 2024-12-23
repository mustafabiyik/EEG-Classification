import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import keras
from keras import layers
import tensorflow as tf
from sklearn import preprocessing, model_selection
import random
import librosa
from util import compute_process_and_plot 
from sklearn.preprocessing import MinMaxScaler
#the samples recorded are given a score from 0 to 128 based on how 
#well-calibrated the sensor was (0 being best, 200 being worst)
QUALITY_THRESHOLD = 128

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 2

eeg = pd.read_csv("eeg-data.csv")


unlabeled_eeg = eeg[eeg["label"] == "unlabeled"]
eeg = eeg.loc[eeg["label"] != "unlabeled"]
eeg = eeg.loc[eeg["label"] != "everyone paired"]

eeg.drop(
    [
        "indra_time",
        "Unnamed: 0",
        "browser_latency",
        "reading_time",
        "attention_esense",
        "meditation_esense",
        "updatedAt",
        "createdAt",
    ],
    axis=1,
    inplace=True,
)

eeg.reset_index(drop=True, inplace=True)
print(eeg.head())



def convert_string_data_to_values(value_string):
    str_list = json.loads(value_string)
    return str_list


eeg["raw_values"] = eeg["raw_values"].apply(convert_string_data_to_values)

eeg = eeg.loc[eeg["signal_quality"] < QUALITY_THRESHOLD]

#visualize one random sample from the data

def view_eeg_plot(idx):
    data = eeg.loc[idx, "raw_values"]
    plt.plot(data)
    plt.title(f"Sample random plot")
    plt.savefig('EEG_example_10.png')

view_eeg_plot(10)

#There are a total of 67 different labels present in the data, where there are numbered
#sub-labels. We collate them under a single label as per their numbering and replace them
#in the data itself. Following this process, we perform simple Label encoding to get them
#in an integer format.


print("Before replacing labels")
print(eeg["label"].unique(), "\n")
print(len(eeg["label"].unique()), "\n")


eeg.replace(
    {
        "label": {
            "blink1": "blink",
            "blink2": "blink",
            "blink3": "blink",
            "blink4": "blink",
            "blink5": "blink",
            "math1": "math",
            "math2": "math",
            "math3": "math",
            "math4": "math",
            "math5": "math",
            "math6": "math",
            "math7": "math",
            "math8": "math",
            "math9": "math",
            "math10": "math",
            "math11": "math",
            "math12": "math",
            "thinkOfItems-ver1": "thinkOfItems",
            "thinkOfItems-ver2": "thinkOfItems",
            "video-ver1": "video",
            "video-ver2": "video",
            "thinkOfItemsInstruction-ver1": "thinkOfItemsInstruction",
            "thinkOfItemsInstruction-ver2": "thinkOfItemsInstruction",
            "colorRound1-1": "colorRound",
            "colorRound1-2": "colorRound",
            "colorRound1-3": "colorRound",
            "colorRound1-4": "colorRound",
            "colorRound1-5": "colorRound",
            "colorRound1-6": "colorRound",
            "colorRound2-1": "colorRound",
            "colorRound2-2": "colorRound",
            "colorRound2-3": "colorRound",
            "colorRound2-4": "colorRound",
            "colorRound2-5": "colorRound",
            "colorRound2-6": "colorRound",
            "colorRound3-1": "colorRound",
            "colorRound3-2": "colorRound",
            "colorRound3-3": "colorRound",
            "colorRound3-4": "colorRound",
            "colorRound3-5": "colorRound",
            "colorRound3-6": "colorRound",
            "colorRound4-1": "colorRound",
            "colorRound4-2": "colorRound",
            "colorRound4-3": "colorRound",
            "colorRound4-4": "colorRound",
            "colorRound4-5": "colorRound",
            "colorRound4-6": "colorRound",
            "colorRound5-1": "colorRound",
            "colorRound5-2": "colorRound",
            "colorRound5-3": "colorRound",
            "colorRound5-4": "colorRound",
            "colorRound5-5": "colorRound",
            "colorRound5-6": "colorRound",
            "colorInstruction1": "colorInstruction",
            "colorInstruction2": "colorInstruction",
            "readyRound1": "readyRound",
            "readyRound2": "readyRound",
            "readyRound3": "readyRound",
            "readyRound4": "readyRound",
            "readyRound5": "readyRound",
            "colorRound1": "colorRound",
            "colorRound2": "colorRound",
            "colorRound3": "colorRound",
            "colorRound4": "colorRound",
            "colorRound5": "colorRound",
        }
    },
    inplace=True,
)

print("After replacing labels")
print(eeg["label"].unique())
print(len(eeg["label"].unique()))

le = preprocessing.LabelEncoder()  #generates a look-up table
le.fit(eeg["label"])
eeg["label"] = le.transform(eeg["label"])

num_classes = len(eeg["label"].unique())
print(num_classes)

class_counts = eeg["label"].value_counts()

class_names = le.inverse_transform(class_counts.index)  #replace with actual class mapping

#create the bar plot
plt.figure(figsize=(12, 6)) 
plt.bar(class_names, class_counts.values, color='skyblue')

plt.title("Number of Samples per Class", fontsize=16)
plt.xlabel("Class Names", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines for clarity

plt.tight_layout()
plt.savefig('EEG_bar.png')



#scale and split data
#We perform a simple Min-Max scaling to bring the value-range between 0 and 1

scaler = preprocessing.MinMaxScaler()
series_list = [
    scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in eeg["raw_values"]
]

labels_list = [i for i in eeg["label"]]

#train test split
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    series_list, labels_list, test_size=0.15, random_state=42, shuffle=True
)

print(
    f"Length of x_train : {len(x_train)}\nLength of x_test : {len(x_test)}\nLength of y_train : {len(y_train)}\nLength of y_test : {len(y_test)}"
)

x_train = np.asarray(x_train).astype(np.float32).reshape(-1, 512, 1)
y_train = np.asarray(y_train).astype(np.float32).reshape(-1, 1)
y_train = keras.utils.to_categorical(y_train)

x_test = np.asarray(x_test).astype(np.float32).reshape(-1, 512, 1)
y_test = np.asarray(y_test).astype(np.float32).reshape(-1, 1)
y_test = keras.utils.to_categorical(y_test)

#create a `tf.data.Dataset` from this data to prepare it for training

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

#As we can see from the plot of number of samples per class, the dataset is imbalanced.
#Hence, we **calculate weights for each class** to make sure that the model is trained in
#a fair manner without preference to any specific class due to greater number of samples.
#We use a naive method to calculate these weights, finding an **inverse proportion** of
#each class and using that as the weight.


vals_dict = {}
for i in eeg["label"]:
    if i in vals_dict.keys():
        vals_dict[i] += 1
    else:
        vals_dict[i] = 1
total = sum(vals_dict.values())

# Formula used - Naive method where
# weight = 1 - (no. of samples present / total no. of samples)
# So more the samples, lower the weight

weight_dict = {k: (1 - (v / total)) for k, v in vals_dict.items()}
print(weight_dict)

#define simple function to plot all the metrics present in a `keras.callbacks.History`


def plot_history_metrics(history: keras.callbacks.History):
    total_plots = len(history.history)
    cols = total_plots // 2

    rows = total_plots // cols

    if total_plots % cols != 0:
        rows += 1

    pos = range(1, total_plots + 1)
    plt.figure(figsize=(15, 10))
    for i, (key, value) in enumerate(history.history.items()):
        plt.subplot(rows, cols, pos[i])
        plt.plot(range(len(value)), value)
        plt.title(str(key))
    plt.savefig('transformer_history_metrics.png')




#define the Transformer model
def transformer_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ffn = layers.Dense(ff_dim, activation="relu")(attention_output)
    ffn = layers.Dense(inputs.shape[-1])(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn)
    return output

def attention_pooling(inputs):
    #learnable weights for attention pooling
    attention_scores = layers.Dense(1, activation="softmax")(inputs)
    weighted_inputs = inputs * attention_scores
    #wrap tf.reduce_sum in a Lambda layer
    return layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_inputs)

def positional_encoding(length, depth):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(depth)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(depth))
    pos_encoding = pos * angle_rates
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])  
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])  
    return tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)

def create_transformer_model():
    input_layer = keras.Input(shape=(512, 1))
    x = layers.Dense(64)(input_layer)
    
    #add positional encoding
    pos_encoding = positional_encoding(512, 64)
    x += pos_encoding

    for _ in range(6):  # Add 3 Transformer blocks
        x = transformer_block(x, num_heads=4, ff_dim=128, dropout_rate=0.1)
    #x = layers.GlobalAveragePooling1D()(x)  # Pool across the time dimension
    # Attention pooling instead of global average pooling
    x = attention_pooling(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output_layer = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)


transformer_model = create_transformer_model()


callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model_transformer.keras", save_best_only=True, monitor="loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_top_k_categorical_accuracy",
        factor=0.2,
        patience=2,
        min_lr=0.000001,
    ),
]



transformer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[
        keras.metrics.TopKCategoricalAccuracy(k=3),
        keras.metrics.AUC(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
    ],
)

transformer_model_history = transformer_model.fit(
    train_dataset,
    epochs=10,
    callbacks=callbacks,
    validation_data=test_dataset,
    class_weight=weight_dict,
)

# Visualize metrics
#plot_history_metrics(transformer_model_history)

# Evaluate the model
loss, accuracy, auc, precision, recall = transformer_model.evaluate(test_dataset)
print(f"Transformer Loss : {loss}")
print(f"Top 3 Categorical Accuracy : {accuracy}")
print(f"Area under the Curve (ROC) : {auc}")
print(f"Precision : {precision}")
print(f"Recall : {recall}")


def view_evaluated_eeg_plots(model):
    start_index = random.randint(10, len(eeg))
    end_index = start_index + 11
    data = eeg.loc[start_index:end_index, "raw_values"]
    data_array = [scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in data]
    data_array = [np.asarray(data_array).astype(np.float32).reshape(-1, 512, 1)]
    original_labels = eeg.loc[start_index:end_index, "label"]
    predicted_labels = np.argmax(model.predict(data_array, verbose=0), axis=1)
    original_labels = [
        le.inverse_transform(np.array(label).reshape(-1))[0]
        for label in original_labels
    ]
    predicted_labels = [
        le.inverse_transform(np.array(label).reshape(-1))[0]
        for label in predicted_labels
    ]
    total_plots = 12
    cols = total_plots // 3
    rows = total_plots // cols
    if total_plots % cols != 0:
        rows += 1
    pos = range(1, total_plots + 1)
    fig = plt.figure(figsize=(20, 10))
    for i, (plot_data, og_label, pred_label) in enumerate(
        zip(data, original_labels, predicted_labels)
    ):
        plt.subplot(rows, cols, pos[i])
        plt.plot(plot_data)
        plt.title(f"Actual Label : {og_label}\nPredicted Label : {pred_label}")
        fig.subplots_adjust(hspace=0.5)
    plt.savefig('trasformer_evaluated_eeg_plots.png')


#view_evaluated_eeg_plots(transformer_model)
