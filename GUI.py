import streamlit as st
import joblib
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from torchvision import transforms

# Load the trained models
svc_model_path = r"C:\Users\sharr\Desktop\SMLPROJECT\svc_model.pkl"
rf_model_path = r"C:\Users\sharr\Desktop\SMLPROJECT\rf_model.pkl"
gbm_model_path = r"C:\Users\sharr\Desktop\SMLPROJECT\gb_model.pkl"
adaboost_model_path = r"C:\Users\sharr\Desktop\SMLPROJECT\adaboost_model.pkl"

svc_model = joblib.load(svc_model_path)
rf_model = joblib.load(rf_model_path)
gbm_model = joblib.load(gbm_model_path)
adaboost_model = joblib.load(adaboost_model_path)

# Initialize Img2Vec model
img2vec = Img2Vec()

# Function to preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(image)
    img_pil = transforms.ToPILImage()(img_tensor)
    img_features = img2vec.get_vec(img_pil)
    return img_features

# Streamlit app
st.image(r"C:\Users\sharr\Desktop\SMLPROJECT\background.jpeg", use_column_width=True)

st.title('Blood Group Prediction')

# Model selection
model_selection = st.selectbox('Select Model', ('Random Forest', 'SVC', 'GBM', 'AdaBoost'))

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction on the uploaded image based on the selected model
    if st.button('Make Prediction'):
        preprocessed_image = preprocess_image(image)
        if model_selection == 'Random Forest':
            prediction = rf_model.predict([preprocessed_image])
            st.write('Prediction (Random Forest):', prediction[0])
        elif model_selection == 'SVC':
            prediction = svc_model.predict([preprocessed_image])
            st.write('Prediction (SVC):', prediction[0])
        elif model_selection == 'GBM':
            prediction = gbm_model.predict([preprocessed_image])
            st.write('Prediction (GBM):', prediction[0])
        elif model_selection == 'AdaBoost':
            prediction = adaboost_model.predict([preprocessed_image])
            st.write('Prediction (AdaBoost):', prediction[0])
