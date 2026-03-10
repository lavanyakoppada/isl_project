### Project Overview

This project presents a real-time bidirectional communication system designed to bridge the communication gap between deaf or mute individuals and people who do not understand sign language. The system detects and recognizes sign language gestures using computer vision and deep learning techniques and converts them into spoken words. Additionally, spoken language from non-sign language users is converted into text so that deaf individuals can understand the conversation.

The goal of this project is to improve accessibility and social inclusion by enabling natural communication without the need for a human interpreter.

### 🚀 Features

  ## Sign to Text/Speech
  Real-time sign language gesture recognition
  
  Conversion of sign language to Text and speech
  
  Live hand landmark detection using MediaPipe
  
  Gesture classification using Artificial Neural Networks
  
  Real-time camera-based gesture detection

## Speech/Text to Sign
  Speech Recognition using Python SpeechRecognition Library
  
  Text to Gloss conversion using NLP and Rule-Based appraoch
  
  Mapping Gloss to Sign Animations

Interactive two-way communication interface

### 🧠 Technologies Used

Python

OpenCV – real-time video processing

MediaPipe – hand landmark detection

NumPy & Pandas – data processing

TensorFlow / Keras – deep learning model

Artificial Neural Network (ANN)

Multi-Layer Perceptron (MLP)

Google Web Speech API – speech to text

gTTS (Google Text-to-Speech) – text to speech

### 🏗 System Architecture
## Sign to Speech

# Gesture Detection

  Webcam captures hand gestures.
  
  MediaPipe extracts hand landmarks.

# Gesture Classification

  Extracted landmarks are fed into a trained ANN/MLP model.
  
  The model predicts the corresponding sign language word.

# Speech Generation

  The predicted word is converted into speech using gTTS.
  
## Speech to Sign

# Speech Input from User

Spoken input is captured using Google Web Speech API.

Speech is converted into text for the deaf user.

# Gloss Generator

Processing and TOkenizatioin of Text using NLP.

Converting tokens into Gloss by following ISL grammer rules.

# Sign Mapper

The Gloss is mapped to the available sign animation dataset.

If the word in gloss doesn't find a match, they will be splitted into alphabets.

### ⚙️ Installation
## 1️⃣ Clone the Repository
git clone https://github.com/yourusername/isl_project.git
cd isl_project

## 2️⃣ Install Dependencies
pip install -r requirements.txt

## 3️⃣ Run the Application
python manage.py runserver

