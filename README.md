# Sentiment-AI

A simple **Sentiment Analysis web application** that classifies user-provided text into three categories: **Positive, Negative, or Neutral**.

The project uses a **PyTorch Bidirectional LSTM (BiLSTM)** model for sentiment classification and a **Flask backend** to serve predictions through an API. A simple frontend allows users to enter text and instantly view the predicted sentiment along with its confidence score.

## Features

* Classifies text as **Positive, Negative, or Neutral**
* PyTorch-based **BiLSTM** sentiment classification model
* Text preprocessing including cleaning, tokenization, and stemming
* Flask REST API for model predictions
* Simple and responsive web interface
* Displays:

  * Predicted sentiment
  * Confidence percentage
  * Sentiment emoji
* Pre-trained model included in the repository
* No model training required to run the application

## Tech Stack

### Machine Learning

* Python
* PyTorch
* BiLSTM
* NumPy
* Scikit-learn

### Backend

* Flask
* Flask-CORS
* REST API

### Frontend

* HTML
* CSS
* JavaScript

## Project Structure

```text
Sentiment-AI/
│
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── requirements.txt
│   │
│   ├── data/
│   │   ├── dataset.csv
│   │   └── generate_dataset.py
│   │
│   └── artifacts/
│       ├── model.pt
│       ├── vocab.json
│       └── config.json
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── README.md
```

## How It Works

The application follows this pipeline:

```text
User Input
    ↓
Text Preprocessing
    ↓
Tokenization
    ↓
Vocabulary Mapping
    ↓
Embedding Layer
    ↓
BiLSTM Model
    ↓
Linear Layer
    ↓
Sentiment Prediction
    ↓
Positive / Neutral / Negative
```

### Model Architecture

The sentiment model consists of:

```text
Input Text
    ↓
Embedding Layer
    ↓
Bidirectional LSTM
    ↓
Linear Layer
    ↓
3-Class Output
```

The model predicts one of the following classes:

| Label | Sentiment |
| ----- | --------- |
| 0     | Negative  |
| 1     | Neutral   |
| 2     | Positive  |

## Dataset

The project includes a small labeled dataset in:

```text
backend/data/dataset.csv
```

The dataset contains **720 labeled text examples** covering Positive, Negative, and Neutral sentiments.

The dataset is primarily intended for demonstrating the complete ML pipeline. For a production-level application, a larger real-world labeled dataset would be recommended.

## Installation

### Prerequisites

Make sure you have:

* Python 3.9 or higher
* Git
* VS Code (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/shrutikushwaha18/Sentiment-AI.git
```

Go to the project directory:

```bash
cd Sentiment-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
venv\Scripts\activate
```

#### Git Bash

```bash
source venv/Scripts/activate
```

### 4. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

## Run the Application

Navigate to the backend directory:

```bash
cd backend
```

Start the Flask server:

```bash
python app.py
```

The application will run at:

```text
http://localhost:5000
```

Open the URL in your browser.

## Example

Enter a sentence such as:

```text
I absolutely loved this product!
```

The application may return:

```text
Sentiment: Positive
Confidence: 98.93%
```

Another example:

```text
The product was disappointing and poor.
```

Possible output:

```text
Sentiment: Negative
```

And:

```text
The package arrived today.
```

Possible output:

```text
Sentiment: Neutral
```

## API

The Flask backend provides a prediction endpoint.

### POST `/api/predict`

Request:

```json
{
  "text": "I absolutely loved this product!"
}
```

Example response:

```json
{
  "label": "Positive",
  "emoji": "😄",
  "confidence": 98.93,
  "probabilities": {
    "Negative": 1.04,
    "Neutral": 0.03,
    "Positive": 98.93
  }
}
```

### GET `/api/health`

This endpoint can be used to check whether the model and backend are running correctly.

## Model Training

A pre-trained model is already included in:

```text
backend/artifacts/
```

Therefore, **training is not required** to run the application.

If you want to retrain the model, run:

```bash
cd backend
python data/generate_dataset.py
python train.py
```

The trained model and related configuration files are saved in:

```text
backend/artifacts/
```

## Model Evaluation

The training pipeline calculates common classification metrics including:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The current prototype achieves approximately **96% validation accuracy** on its held-out synthetic dataset.

> Note: This accuracy is based on the included dataset and should not be treated as an indicator of real-world performance. A larger, real-world labeled dataset would be required for a production application.

## Future Improvements

Possible improvements include:

* Training on larger real-world sentiment datasets
* Improving text preprocessing
* Using pretrained NLP models such as BERT
* Adding multilingual sentiment analysis
* Improving model evaluation with cross-validation
* Deploying the application to a cloud platform

## Project Objective

The objective of this project is to demonstrate an **end-to-end Natural Language Processing and Machine Learning pipeline** where user text is preprocessed, passed through a trained BiLSTM model, and classified into Positive, Negative, or Neutral sentiment.
