import pandas as pd
import numpy as np


# One hot encoding ---> Convert the integer labels (digits 0–9) into a one-hot encoded matrix...
def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot


#Loading and Preprocessing Data module....

def load_data(train_csv, test_csv):
    
    #CSV Loading:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    #Data Separation for train,test and input_feature and output_feature:
    X_train = train_df.drop("label", axis=1).values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int32)
    X_test = test_df.drop("label", axis=1).values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int32)
    
    #Normalization: feature scalling of pixel values from 0-255 to 0 to 1........
    X_train /= 255.0
    X_test /= 255.0
    
    #One-Hot Encoding:Converts labels into one-hot encoded vectors.....
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)

    # Transposes the data so that each sample is a column (matching the network’s expected input shape).
    X_train_T = X_train.T
    X_test_T  = X_test.T
    Y_train_T = y_train_encoded.T
    Y_test_T  = y_test_encoded.T
    
    return X_train_T, Y_train_T, X_test_T, Y_test_T

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data("data/mnist_train.csv", "data/mnist_test.csv")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    
#______________________________________________________________________________end_______________________________________
