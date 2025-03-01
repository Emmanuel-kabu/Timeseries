import os
import shutil
import pathlib
import tensorflow as tf
from Text import (
    download_and_extract,
    clean_dataset,
    split_dataset,
    load_dataset,
    preprocess_dataset,
    build_and_train_model,
    evaluate_model
)

def test_download_and_extract():
    download_and_extract()
    assert os.path.exists("aclImdb"), "Dataset was not extracted successfully."

def test_clean_dataset():
    clean_dataset()
    assert not os.path.exists("aclImdb/train/unsup"), "Unsupervised data was not removed."

def test_split_dataset():
    split_dataset()
    assert os.path.exists("aclImdb/val"), "Validation directory was not created."

def test_load_dataset():
    train_ds, val_ds, test_ds = load_dataset()
    assert train_ds is not None, "Training dataset not loaded."
    assert val_ds is not None, "Validation dataset not loaded."
    assert test_ds is not None, "Test dataset not loaded."

def test_preprocess_dataset():
    train_ds, val_ds, test_ds = load_dataset()
    binary_train_ds, binary_val_ds, binary_test_ds = preprocess_dataset(train_ds, val_ds, test_ds)
    assert binary_train_ds is not None, "Training dataset preprocessing failed."

def test_build_and_train_model():
    train_ds, val_ds, _ = load_dataset()
    binary_train_ds, binary_val_ds, _ = preprocess_dataset(train_ds, val_ds, _)
    model = build_and_train_model(binary_train_ds, binary_val_ds)
    assert isinstance(model, tf.keras.Model), "Model training failed."

def test_evaluate_model():
    train_ds, val_ds, test_ds = load_dataset()
    binary_train_ds, binary_val_ds, binary_test_ds = preprocess_dataset(train_ds, val_ds, test_ds)
    model = build_and_train_model(binary_train_ds, binary_val_ds)
    evaluate_model(model, binary_test_ds)

if __name__ == "__main__":
    pytest.main()
