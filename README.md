# 🕵️‍♂️ Deepfake Detection System

A **Deepfake Detection System** that identifies whether a video is **real** or **fake** using **Deep Learning** techniques.  
This was developed as a **Mini Project** during our **6th semester B.Tech (CSE - AI & ML)**.

---

## 🌟 Features
- Detects **deepfake videos** with good accuracy.  
- Simple and lightweight **Flask-based web interface**.  
- Clear and modular code structure for easy understanding.  
- Built for learning and demonstrating deepfake detection concepts.

---

## ⚠️ Important Note
> **This repository only includes the code for detecting deepfakes in video files (`.mp4`, `.avi`, etc.).**

The version that supports **image (`.jpg`) uploads** is **not included** here.  
If you need the image detection version, you can **contact me directly** —  
I can provide it for an **additional cost**. 😁

---

## 🏗️ Tech Stack

| **Category**         | **Technology**              |
|----------------------|------------------------------|
| **Frontend**         | HTML, CSS, Bootstrap         |
| **Backend**          | Python (Flask)               |
| **Deep Learning**    | TensorFlow / Keras           |
| **Model**            | EfficientNet                 |
| **Version Control**  | Git & GitHub                  |

---

## 📂 Project Structure

Deepfake-Detection/
│
├── app/ # Flask routes, templates, utilities
│ ├── templates/ # HTML files
│ └── static/ # CSS, JS, and images
│
├── model/ # Trained deep learning model (not pushed to GitHub)
├── uploads/ # Uploaded files for detection (ignored by Git)
│
├── config.py # Configuration settings
├── run.py # Start Flask web server
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## ⚙️ Installation & Setup

Follow these steps to run the project on your local machine.

### **1. Clone the Repository**
```bash
git clone https://github.com/Abinsha1221/Deepfake-Detection.git
cd Deepfake-Detection
```
### **2.Create a Virtual Environment (Recommended)**
```
python -m venv venv
venv\Scripts\activate   # For Windows
source venv/bin/activate # For Mac/Linux
```
### **3. Install Dependencies**
```
pip install -r requirements.txt
```
### **4. Add the Trained Model**
```
Place your trained EfficientNet model file inside the model/ directory.

Note: The trained model is not included in this repository due to size limits.
```
▶️ Usage
```
Run the Flask Application
python run.py
```
Once the server starts, open your browser and go to:
```
http://127.0.0.1:5000
```
---
When you're done adding this, I'll send the **final part** with *training instructions, future enhancements, and contact info*. ✅

