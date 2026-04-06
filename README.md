# 🧠 Medical AI Dashboard

## 📌 Overview

This project is a **multi-modal AI system** that combines deep learning and machine learning to assist in medical diagnosis.

It performs:

* 🧠 Brain tumor detection from MRI images (CNN)
* ❤️ Heart disease prediction using patient data (ML)
* 🔥 Explainable AI using Grad-CAM

---

## 🚀 Features

* Upload MRI scan and get tumor classification
* Enter patient details for heart disease prediction
* Grad-CAM visualization to highlight important regions
* Interactive web app using Streamlit
* Clean and user-friendly UI

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Scikit-learn
* Streamlit
* OpenCV
* Matplotlib / Seaborn

---

## 📊 Model Details

### 🧠 MRI Model (CNN)

* Input: MRI image
* Output: Tumor type (glioma, meningioma, no tumor, pituitary)
* Improvements:

  * Data augmentation
  * Dropout layer
* Accuracy: ~86%

---

### ❤️ Tabular Model (Random Forest)

* Input: Patient health data
* Output: Heart disease risk
* Accuracy: ~98%

---

## 🔍 Explainable AI

This project uses **Grad-CAM** to visualize which regions of MRI images influence model predictions, improving interpretability.

---

## 📷 Demo

<img width="1893" height="677" alt="image" src="https://github.com/user-attachments/assets/b4700c45-b452-49c9-be54-12424a2bfef5" />
<img width="660" height="739" alt="image" src="https://github.com/user-attachments/assets/d4b88143-f18a-4505-929d-67e10c4d378f" />


---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 📁 Project Structure

```
Medical_AI_System/
├── app/
│   └── app.py
├── models/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 💡 Future Improvements

* Deploy as a web service
* Use advanced models like ResNet
* Integrate unified medical dataset

---

## 👨‍💻 Author

Jeffrin Merino J
