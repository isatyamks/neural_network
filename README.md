# Neural Network from Scratch

A simple yet powerful feedforward neural network built from scratch using only **NumPy**. This project trains on the **MNIST dataset** (handwritten digits) and can also classify custom **28×28 images**.

---

## 📌 Table of Contents

- [📖 Overview](#-overview)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#-installation)
- [🚀 Usage](#-usage)
  - [1️⃣ Training the Model](#1️⃣-training-the-model)
  - [2️⃣ Testing on MNIST Test Set](#2️⃣-testing-on-mnist-test-set)
  - [3️⃣ Testing with Custom Images](#3️⃣-testing-with-custom-images)
- [🧠 How It Works](#-how-it-works)
  - [Network Architecture](#network-architecture)
  - [Forward Propagation](#forward-propagation)
  - [Backpropagation](#backpropagation)
  - [Parameter Updates](#parameter-updates)
- [📌 Code Files](#-code-files)
- [📜 License](#-license)

---

## 📖 Overview

✅ **Goal:** Classify handwritten digits (0–9) using a neural network built from scratch.
✅ **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/) (60,000 training, 10,000 test images).
✅ **Testing:** Custom 28×28 **JPEG** digit images stored in the `image/` directory.

🖥️ **Neural Network Architecture:**
- **Input Layer:** 784 neurons (one per pixel in a 28×28 image).
- **Hidden Layer:** 64 neurons (configurable) using **ReLU activation**.
- **Output Layer:** 10 neurons (one per digit, 0-9) using **softmax activation**.
- **Loss Function:** Cross-Entropy Loss.
- **Optimizer:** Mini-batch Gradient Descent.

---

## 📁 Project Structure

```
NEURAL_NETWORK/
├── data/
│   ├── test_images/          # Custom test images
│   ├── mnist_test.csv        # MNIST test dataset
│   └── mnist_train.csv       # MNIST training dataset
├── model/
│   └── nn_mnist_model.pkl    # Saved trained model
├── neural_network/
│   ├── nn_core/
│   │   ├── backpropagation.py
│   │   ├── forward_propagation.py
│   │   ├── params_update.py
│   │   └── train_loop.py
│   ├── data_preprocessing.py
│   ├── nn_class.py
│   ├── nn_testing.py
│   ├── nn_training.py
│   └── utils.py
├── notebook/
│   └── nn.ipynb              # Jupyter Notebook
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 🔹 Step 1: Clone the Repository
```bash
git clone https://github.com/isatyamks/neural_network.git
cd neural_network
```

### 🔹 Step 2: Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 🔹 Step 3: Install Dependencies
```bash
pip install numpy pandas matplotlib
```
*(Add `Pillow` if working with images.)*

---

## 🚀 Usage

### 1️⃣ Training the Model
✅ Ensure MNIST dataset (`mnist_train.csv`, `mnist_test.csv`) is in the `data/` folder.
✅ Run the training script:
```bash
python train.py
```

📌 **What this does:**
- Loads MNIST data.
- Initializes and trains the neural network.
- Evaluates accuracy on the MNIST test set.
- Saves the trained model to `model/nn_mnist_model.pkl`.
- Plots training and validation loss.

---

### 2️⃣ Testing on MNIST Test Set
Once trained, the script will **automatically** evaluate the model on `mnist_test.csv` and print accuracy.

---

### 3️⃣ Testing with Custom Images
Want to test on **your own** handwritten digits?

✅ Save 28×28 **JPEG** images in `image/`.
✅ Run the script:
```bash
python testing.py
```

📌 **What this does:**
- Loads the trained model (`nn_mnist_model.pkl`).
- Reads each image from `image/`, converts to a NumPy array (28×28).
- Normalizes, flattens, and predicts the digit class.

💡 *Modify `testing.py` to display the image alongside its predicted label!*

---

## 🧠 How It Works

### 🔹 Network Architecture
| Layer          | Details                          |
|---------------|---------------------------------|
| **Input**     | 28×28 = 784 neurons             |
| **Hidden**    | 64 neurons, ReLU activation    |
| **Output**    | 10 neurons, Softmax activation |

---

### 🔹 Forward Propagation
1️⃣ **Compute Weighted Sum:**
   \[ Z = W \times X + b \]
2️⃣ **Apply Activation Function:**
   - Hidden Layer: **ReLU** (max(0, Z))
   - Output Layer: **Softmax** (probability distribution over 10 digits)

---

### 🔹 Backpropagation
1️⃣ Compute **Loss** (Cross-Entropy for classification).
2️⃣ Calculate **Gradients** using the **Chain Rule**.
3️⃣ Update **Weights & Biases** (Mini-batch Gradient Descent):
   \[ W = W - \alpha \times \text{gradient} \]

---

### 🔹 Parameter Updates
✅ **Gradient Descent** minimizes loss by adjusting weights using `params_update.py`.
✅ Adjust **learning rate** and **batch size** for better performance.

---

## 📌 Code Files

🔹 **Neural Network Class:** [`nn_class.py`](#)
🔹 **Training Script:** [`nn_training.py`](#)
🔹 **Testing Script:** [`nn_testing.py`](#)
🔹 **Data Preprocessing:** [`data_preprocessing.py`](#)
🔹 **Core Modules (Forward, Backward, Update):** [`nn_core/`](#)

---

## 📜 License
This project is open-source under the **MIT License**. Feel free to use, modify, and improve it!

🚀 **Happy Learning & Coding!** 💡

