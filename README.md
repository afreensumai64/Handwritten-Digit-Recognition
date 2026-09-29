# Handwritten Digit Recognition using CNN

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features

- CNN-based handwritten digit classification
- 99%+ accuracy on the MNIST test dataset
- Real-world handwritten image preprocessing
- Multi-digit recognition
- Individual digit prediction with confidence scores
- Streamlit web application

## Tech Stack

Python • TensorFlow/Keras • NumPy • Pandas • Streamlit

## Demo

### Application Output 1

![Handwritten Digit Recognition - Output 1](Screenshots/img1.png)

### Application Output 2

![Handwritten Digit Recognition - Output 2](Screenshots/img2.png)

### Application Output 3

![Handwritten Digit Recognition - Output 3](Screenshots/img3.png)

## How to Run

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd Handwritten-Digit-Recognition
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application will open in your browser.


## Project Structure

```text
Handwritten-Digit-Recognition/
│
├── app/
│   └── app.py
│
├── models/
│   └── handwritten_digit_cnn.keras
│
├── notebooks/
│   └── Handwritten_Character_Recognition.ipynb
│
├── outputs/
│   ├── plots/
│   └── results/
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── Screenshots/
│   ├── img1.png
│   ├── img2.png
│   └── img3.png
│
├── README.md
└── requirements.txt
```
## Applications

- **Banking & Cheque Processing** – Recognizing handwritten numbers in cheques and financial forms.
- **Form Digitization** – Converting handwritten numerical entries into digital data.
- **Postal Processing** – Recognizing handwritten PIN/ZIP codes and numerical addresses.
- **Document Processing** – Extracting handwritten numbers from scanned documents.
- **Data Entry Automation** – Reducing manual effort when converting handwritten numerical information into machine-readable data.
- **Educational Tools** – Supporting automated recognition of handwritten numerical answers.

