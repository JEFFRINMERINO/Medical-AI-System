import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np

# ---------------- MODEL ----------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
def generate_gradcam(model, image, target_layer):
        gradients = []
        activations = []

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        def forward_hook(module, inp, out):
            activations.append(out)

    
            # register hooks
        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_backward_hook(backward_hook)

        output = model(image)
        pred_class = output.argmax(dim=1)

        model.zero_grad()
        output[0, pred_class].backward()

        grads = gradients[0]
        acts = activations[0]

        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1)

        cam = cam.squeeze().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        # resize to image size
        cam = cv2.resize(cam, (128,128))

        handle_f.remove()
        handle_b.remove()

        return cam

import pickle

model_tab = pickle.load(open(r"D:\Medical_AI_System\models\tabular_model_v2.pkl", "rb"))

# ---------------- LOAD MODEL ----------------
model = CNNModel()
model.load_state_dict(torch.load(r"D:\Medical_AI_System\models\mri_model_v2.pth", map_location=torch.device('cpu')))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------------- UI ----------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Medical AI Dashboard</h1>", unsafe_allow_html=True)
st.sidebar.title("📊 About App")

st.sidebar.write("""
This Medical AI System uses:
- CNN for MRI tumor detection
- Machine Learning for heart disease prediction
- Grad-CAM for explainability
""")
st.sidebar.title("📌 Instructions")

st.sidebar.write("""
1. Enter patient details  
2. Upload MRI image  
3. Click Analyze  
4. View results  
""")


col1, col2 = st.columns(2)

with col2:
    st.subheader("Enter Patient Details")

age = st.number_input("Age", 1, 100, 30)
sex = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0
cp = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)

cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox(
    "Fasting Blood Sugar",
    ["Normal (<=120)", "High (>120)"]
)

fbs = 1 if fbs == "High (>120)" else 0
restecg = st.selectbox(
    "Resting ECG Result",
    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)

restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox(
    "Exercise-Induced Chest Pain",
    ["No", "Yes"]
)

exang = 1 if exang == "Yes" else 0
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox(
    "Slope of Peak Exercise",
    ["Upsloping", "Flat", "Downsloping"]
)

slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
ca = st.selectbox("CA (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox(
    "Thalassemia Type",
    ["Normal", "Fixed Defect", "Reversible Defect"]
)

thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

with col1:
    st.subheader("🧠 MRI Upload")
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])
    predict_button = st.button("🔍 Analyze")

    if "run_prediction" not in st.session_state:
        st.session_state.run_prediction = False

    if predict_button:
        st.session_state.run_prediction = True

import numpy as np

tab_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

tab_pred = model_tab.predict(tab_input)

st.subheader("❤️ Heart Disease Prediction")

if tab_pred[0] == 1:
    st.error("High Risk")
else:
    st.success("Low Risk")

if st.session_state.run_prediction and uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=300)
    

    # preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Grad-CAM
    cam = generate_gradcam(model, img_tensor, model.conv2)

    # original image (for plotting)
    img_np = img_tensor.squeeze().numpy()

    # overlay heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0

    overlay = heatmap + np.stack([img_np]*3, axis=-1)
    overlay = overlay / overlay.max()

    st.subheader("🔥 Grad-CAM Visualization")
    st.image(overlay, caption="🔥 Grad-CAM Heatmap", width = 500)
    # preprocess
    image = transform(image)
    image = image.unsqueeze(0)
    
    # prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Tumor Type", classes[predicted.item()])

    with col2:
        if tab_pred[0] == 1:
            st.metric("Heart Risk", "High ⚠️")
        else:
            st.metric("Heart Risk", "Low ✅")

    
    st.subheader("🧠 Brain Tumor Prediction")
    st.success(f"{classes[predicted.item()]}")



