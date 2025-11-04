# Optical Character Recognition (OCR) System 

This project is a Convolutional Neural Network (CNN)-based Optical Character Recognition (OCR) system developed using **Python** and **TensorFlow**. It enables recognition of alphanumeric characters (A–Z, 0–9) from grayscale images and provides predictions via a simple **Tkinter GUI**.

##  Features
- Character recognition using a CNN trained on a standard OCR dataset
- Real-time image input through GUI
- Accuracy: **98.81%** on test dataset
- Supports digits (0–9) and uppercase letters (A–Z)

##  Objectives
- Build an OCR system using CNN and TensorFlow
- Train the model to recognize characters from input images
- Evaluate accuracy and usability with real-world inputs

##  Technologies Used
- **Language**: Python  
- **Libraries**: TensorFlow (Keras), OpenCV, Tkinter  
- **Model**: CNN with 3 convolutional layers, ReLU activations, dropout, and softmax output

##  Dataset
- [Standard OCR Dataset (Kaggle)](https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset)  
- Contains 64×64 grayscale images of digits and uppercase letters.

##  Model Architecture
- **Input**: 64×64 grayscale images
- **Layers**:
  - 3× Conv2D + ReLU + MaxPooling
  - Flatten + Dense (with Dropout)
  - Dense Softmax (36 classes)

##  GUI Interface
- Select and preview image
- View predicted character below the image
- Instant prediction on new input

##  Results
- **Training Accuracy**: 97%
- **Test Accuracy**: 98.81%
- Minimal misclassification, mainly among visually similar characters (e.g., ‘O’ vs ‘0’)

##  Future Enhancements
- Full word/sentence recognition using CRNN or RNN+CTC
- Multilingual support (e.g., Devanagari, Arabic)
- Mobile or web deployment (Flask, Streamlit, Flutter)
- Improved handwriting recognition

##  Contributors
- Dikshyant Adhikari   

##  References
- LeNet-5 (LeCun et al.)
- Tesseract OCR Engine (Google)
- Deep Learning (Goodfellow et al.)
- [Kaggle Dataset](https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset)

---

*"Making machines read like humans – one pixel at a time."*
