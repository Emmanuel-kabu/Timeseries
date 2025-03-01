import os
import pathlib
import shutil
import random
import subprocess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
# Download and extract dataset
def download_and_extract():
    dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    if not os.path.exists("aclImdb_v1.tar.gz"):
        subprocess.run(["curl", "-O", dataset_url], check=True)
    if not os.path.exists("aclImdb"):
        subprocess.run(["tar", "-xf", "aclImdb_v1.tar.gz"], check=True)

# Remove unsupervised training data
def clean_dataset():
    unsup_path = "aclImdb/train/unsup"
    if os.path.exists(unsup_path):
        shutil.rmtree(unsup_path)

# Split data into training and validation sets
def split_dataset():
    base_dir = pathlib.Path("aclImdb")
    val_dir = base_dir / "val"
    train_dir = base_dir / "train"
    os.makedirs(val_dir, exist_ok=True)
    
    for category in ("neg", "pos"):
        os.makedirs(val_dir / category, exist_ok=True)
        files = os.listdir(train_dir / category)
        random.Random(1337).shuffle(files)
        num_val_samples = int(0.2 * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            shutil.move(train_dir / category / fname, val_dir / category / fname)

# Load dataset
def load_dataset(batch_size=32):
    train_ds = keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=batch_size)
    val_ds = keras.utils.text_dataset_from_directory("aclImdb/val", batch_size=batch_size)
    test_ds = keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=batch_size)
    return train_ds, val_ds, test_ds

# Preprocess dataset
def preprocess_dataset(train_ds, val_ds, test_ds):
    text_vectorization = TextVectorization(max_tokens=20000, output_mode='multi_hot')
    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)
    
    binary_1gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
    binary_1gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
    binary_1gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
    
    return binary_1gram_train_ds, binary_1gram_val_ds, binary_1gram_test_ds

# Build and train model
def build_and_train_model(train_ds, val_ds):
    inputs = keras.Input(shape=(20000,))
    x = layers.Dense(16, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = [keras.callbacks.ModelCheckpoint("binary_1gram.keras", save_best_only=True)]
    
    model.fit(train_ds.cache(), validation_data=val_ds.cache(), epochs=10, callbacks=callbacks)
    return model

# Evaluate model
def evaluate_model(model, test_ds):
    model = keras.models.load_model("binary_1gram.keras")
    test_acc = model.evaluate(test_ds)[1]
    print(f"Test Accuracy: {test_acc:.3f}")

if __name__ == "__main__":
    download_and_extract()
    clean_dataset()
    split_dataset()
    
    train_ds, val_ds, test_ds = load_dataset()
    binary_1gram_train_ds, binary_1gram_val_ds, binary_1gram_test_ds = preprocess_dataset(train_ds, val_ds, test_ds)
    
    model = build_and_train_model(binary_1gram_train_ds, binary_1gram_val_ds)
    evaluate_model(model, binary_1gram_test_ds)







