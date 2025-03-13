import pandas as pd
import numpy as np

def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

def load_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    X_train = train_df.drop("label", axis=1).values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int32)
    X_test = test_df.drop("label", axis=1).values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int32)
    
    X_train /= 255.0
    X_test /= 255.0
    
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)
    
    return X_train.T, y_train_encoded.T, X_test.T, y_test_encoded.T

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data("data/mnist_train.csv", "data/mnist_test.csv")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
