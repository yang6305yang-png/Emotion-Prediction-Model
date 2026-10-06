# 🧠 End-to-End NLP Emotion Classification with Deep Learning, FastAPI & Deployment

An end-to-end Natural Language Processing (NLP) project that classifies text into **six different emotions** using Deep Learning. The project covers the complete workflow from dataset loading and NLP preprocessing to model comparison, evaluation, model serialization, and API/deployment integration.

## 🚀 Project Overview

The goal of this project is to build a Deep Learning based **Emotion Classification** system that takes a piece of text as input and predicts its emotional category.

The project uses the **DAIR-AI Emotion dataset** from Hugging Face and experiments with multiple recurrent neural network architectures before selecting a **Bidirectional GRU (BiGRU)** model as the final model.

### 🎯 Supported Emotions

- 😢 Sadness
- 😊 Joy
- ❤️ Love
- 😠 Anger
- 😨 Fear
- 😲 Surprise

---

## 🏗️ End-to-End Workflow

```text
Hugging Face Emotion Dataset
            ↓
     Data Exploration
            ↓
    NLP Preprocessing
            ↓
      Tokenization
            ↓
     Padding Sequences
            ↓
   Deep Learning Models
   ├── Simple RNN
   ├── LSTM
   ├── GRU
   └── Bidirectional GRU
            ↓
     Model Comparison
            ↓
       BiGRU Selection
            ↓
      Model Evaluation
            ↓
   Save Model + Tokenizer
            ↓
        FastAPI API
            ↓
        Deployment
```

---

## 📊 Dataset

The project uses the **`dair-ai/emotion`** dataset from Hugging Face.

| Split | Samples |
|---|---:|
| Training | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |
| **Total** | **20,000** |

The dataset contains two main fields:

- `text` — input text
- `label` — emotion class

The six emotion labels are:

```text
sadness, joy, love, anger, fear, surprise
```

---

## 🔎 Exploratory Data Analysis

The notebook performs basic EDA to understand:

- Dataset structure
- Missing values
- Emotion/class distribution
- Sample text data
- Label frequencies

No missing values were found in the training text/label columns during the notebook analysis.

---

## 🧹 NLP Preprocessing

The following preprocessing pipeline is implemented:

1. Text tokenization using Keras `Tokenizer`
2. Vocabulary configuration with `num_words = 10,000`
3. Out-of-vocabulary token: `<unk>`
4. Conversion of text into integer sequences
5. Sequence padding/truncation
6. Maximum sequence length: **50**

The notebook reports a tokenizer vocabulary index size of **15,213**, while the model input vocabulary is limited using `num_words=10,000`.

---

## 🤖 Deep Learning Models

Four recurrent neural network approaches were explored.

### 1. Simple RNN

```text
Embedding
   ↓
SimpleRNN(128)
   ↓
Dropout(0.5)
   ↓
SimpleRNN(64)
   ↓
Dropout(0.5)
   ↓
Dense + Softmax
```

### 2. LSTM

```text
Embedding
   ↓
LSTM(128)
   ↓
Dropout(0.5)
   ↓
LSTM(64)
   ↓
Dropout(0.5)
   ↓
Dense + Softmax
```

### 3. GRU

```text
Embedding
   ↓
GRU(128)
   ↓
Dropout(0.5)
   ↓
GRU(64)
   ↓
Dropout(0.5)
   ↓
Dense + Softmax
```

### 4. Bidirectional GRU — Final Model ⭐

```text
Input (50 tokens)
       ↓
Embedding (300 dimensions)
       ↓
Bidirectional GRU (128)
       ↓
Dropout (0.5)
       ↓
Bidirectional GRU (64)
       ↓
Dropout (0.5)
       ↓
Dense + Softmax
       ↓
Emotion Prediction
```

The BiGRU model was selected as the final model because it achieved the strongest test accuracy among the models evaluated in the notebook.

---

## ⚙️ Training Configuration

Common training components include:

- Optimizer: **Adam**
- Loss: **Sparse Categorical Crossentropy**
- Batch size: **32**
- Maximum epochs: **20**
- Early stopping based on validation loss
- `restore_best_weights=True`
- Class weights for handling class imbalance
- Softmax output layer for multi-class classification

---

## 📈 Model Performance

The notebook reports the following test results:

| Model | Test Loss | Test Accuracy |
|---|---:|---:|
| Simple RNN | 1.7386 | 33.45% |
| LSTM | 0.3687 | 88.25% |
| GRU | 1.7779 | 30.90% |
| **Bidirectional GRU** | **0.2221** | **92.15%** |

### 🏆 Best Model

**Bidirectional GRU (BiGRU)**

- Test Loss: **0.2221**
- Test Accuracy: **92.15%**

The BiGRU model was therefore used for the final prediction and model serialization steps.

---

## 🧪 Sample Predictions

The trained BiGRU model was tested on sample sentences.

| Input | Predicted Emotion |
|---|---|
| “I can't believe how happy I am right now, this is amazing!” | Surprise |
| “I feel so alone and hopeless today.” | Sadness |
| “I am furious that they cancelled the trip at the last minute.” | Anger |
| “I feel terrified when walking down dark alleyways alone.” | Fear |
| “I was shocked and completely surprised by the unexpected gift!” | Surprise |

These examples demonstrate the model's end-to-end prediction pipeline from raw text to emotion label.

---

## 📊 Model Evaluation

The final BiGRU model is evaluated using:

- Test loss
- Test accuracy
- Confusion matrix
- Sample text predictions

A confusion matrix is generated to analyze class-wise prediction performance across the six emotion categories.

---

## 💾 Model & Tokenizer Saving

The trained model and tokenizer are saved inside the `Artifacts` directory.

```text
Artifacts/
├── BiGRU_Modle.keras
└── tokenizer.pkl
```

> Note: `BiGRU_Modle.keras` is the filename currently used by the project.

The tokenizer is serialized using Python `pickle`, allowing the same preprocessing pipeline to be reused during inference.

---

## ⚡ FastAPI Integration

The repository also contains a `main.py` file for serving the trained NLP model through **FastAPI**.

The API layer is intended to provide a way for an external client/frontend to send text and receive the predicted emotion without directly interacting with the training notebook.

Typical architecture:

```text
Client / Frontend
       ↓
    FastAPI
       ↓
Tokenizer
       ↓
   BiGRU Model
       ↓
Emotion Prediction
       ↓
    API Response
```

For the exact request/response schema and endpoint implementation, refer to `main.py`.

---

## 🌐 Deployment

The repository includes deployment-related configuration such as:

- `requirements.txt`
- `runtime.txt`
- `main.py`

This allows the trained model and FastAPI application to be prepared for cloud deployment.

---

## 📁 Project Structure

```text
NLP-Project/
│
├── Artifacts/
│   ├── BiGRU_Modle.keras
│   └── tokenizer.pkl
│
├── static/
│
├── NLP Project with Deep Learning.ipynb
├── main.py
├── requirements.txt
├── runtime.txt
├── README.md
└── .gitignore
```

`__pycache__/` should ideally be excluded from version control using `.gitignore`.

---

## 🛠️ Tech Stack

### Programming Language
- Python

### NLP & Data Processing
- Pandas
- NumPy
- Hugging Face Datasets
- TensorFlow / Keras

### Machine Learning
- Scikit-learn

### Deep Learning
- Simple RNN
- LSTM
- GRU
- Bidirectional GRU

### API
- FastAPI

### Visualization
- Matplotlib
- Seaborn

### Deployment
- FastAPI deployment setup
- `requirements.txt`
- `runtime.txt`

---

## ▶️ Running the Project Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI application

If the FastAPI application object in `main.py` is named `app`:

```bash
uvicorn main:app --reload
```

The exact API endpoint and input format are defined in `main.py`.

---

## 💡 Key Learnings

Through this project, I worked on:

- Building an end-to-end NLP pipeline
- Loading and exploring a real-world emotion dataset
- Text tokenization and sequence padding
- Handling class imbalance with class weights
- Building and comparing RNN architectures
- Training LSTM, GRU and BiGRU models
- Using EarlyStopping
- Evaluating Deep Learning models
- Creating confusion matrices
- Saving trained models and tokenizers
- Integrating a trained NLP model with FastAPI
- Preparing an ML application for deployment

---

## 🔮 Future Improvements

Possible improvements include:

- Use a dedicated validation set during model development and reserve the test set strictly for final evaluation
- Add precision, recall and F1-score for every emotion class
- Add a classification report
- Improve error analysis for misclassified emotions
- Experiment with pretrained Transformer models such as BERT
- Add API input validation and structured response schemas
- Add automated tests for the API
- Add a frontend interface for interactive emotion prediction
- Add CI/CD for automated deployment

---

## 👨‍💻 Project

**End-to-End NLP Emotion Classification with Deep Learning, FastAPI & Deployment**

This project demonstrates how an NLP Deep Learning model can be taken from **data preprocessing → model training → evaluation → serialization → API integration → deployment** as a complete machine learning application.
