# 🧠 FFNN from Scratch - Major Assignment I IF3270 Machine Learning

This repository contains an implementation of a **Feedforward Neural Network (FFNN)** from scratch (without deep learning libraries like TensorFlow or PyTorch), developed as part of the **Machine Learning course major assignment I**.

## 📁 Project Structure

```
tubes-1-ml-main/
├── model/                  # Output folder for saved models (.npy)
├── src/                   # Main source code folder
│   ├── ffnn.py            # FFNN class implementation from scratch
│   └── tests.ipynb        # Jupyter notebook for experiments and testing
├── .gitignore
```

## 🚀 Key Features

- Flexible architecture (custom number of layers and neurons)
- Supports multiple activation functions: `sigmoid`, `relu`, `tanh`, `softmax`, `linear`
- Loss functions: `mse`, `binary_crossentropy`, `categorical_crossentropy`
- Weight initialization options: `zero`, `random_uniform`, `random_normal`
- Visualization:
  - Network structure with weights
  - Weight and gradient distributions
  - Training and validation loss curves
- Mini-batch gradient descent training

## 📊 Experiments

All experiments are included in `tests.ipynb` and cover:

1. **Effect of Architecture (Depth & Width)**
2. **Effect of Activation Functions**
3. **Effect of Learning Rate**
4. **Effect of Weight Initialization**
5. **Comparison with `MLPClassifier` from `sklearn`**

Each experiment includes visualizations such as loss plots, accuracy, histogram of weights and gradients, and confusion matrices.

## 🧪 How to Run

### 1. Requirements
Make sure you have Python 3 and the following libraries:

```bash
pip install numpy matplotlib networkx scikit-learn
```

### 2. Running the Notebook
Open `src/tests.ipynb` using Jupyter Notebook or JupyterLab and run all cells step by step.

## 💡 Notes

- The dataset used is **MNIST digit classification**, with input shape `784` and output of `10` classes (one-hot encoded).
- The model can be saved and loaded using `.save_model()` and `.load_model()` methods.

## 👨‍💻 Contributor

<!-- Buatkan tabel 3 kolom -->
- 13521028 | Muhammad Zulfiansyah Bayu Pratama
- 13521031 | Fahrian Afdholi
- 13521049 | Brian Kheng