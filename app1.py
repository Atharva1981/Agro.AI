import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
# Import preprocess_input from the correct submodule
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image
import io
import os

# Load model and data
@st.cache_resource
def load_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "shuffuled_model.h5")
    df_path = os.path.join(base_dir, "shuffled_file.csv")
    supplement_path = os.path.join(base_dir, "supplement_info.csv")
    info_path = os.path.join(base_dir, "disease_info.csv")

    model = load_model(model_path)
    df = pd.read_csv(df_path)
    le = LabelEncoder()
    le.fit(df['label'].values)
    supplement_df = pd.read_csv(supplement_path)
    info_df = pd.read_csv(info_path, encoding='latin-1')
    return model, df, le, supplement_df, info_df

model, df, le, supplement_df, info_df = load_all()


feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Title
st.title("🌿 AI Plant Disease Detection")

# Upload
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, use_column_width=True)
    
    # Preprocess
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)[0].reshape(1, -1)
    
    X_train_features = df.drop(columns=['image_name', 'label']) if 'image_name' in df.columns else df.drop(columns=['label'])
    scaler = StandardScaler()
    scaler.fit(X_train_features)
    features_scaled = scaler.transform(features)
    features_scaled = np.expand_dims(features_scaled, axis=2)
    
    prediction = model.predict(features_scaled)
    predicted_index = np.argmax(prediction)
    predicted_label = le.inverse_transform([predicted_index])[0]

    st.markdown(f"### 🩺 Predicted Disease: `{predicted_label}`")

    # Normalize
    def normalize_name(name):
        return str(name).strip().lower().replace(" ", "").replace(":", "").replace("|", "").replace("_", "")
    
    predicted_clean = normalize_name(predicted_label)
    supplement_df['normalized'] = supplement_df['disease_name'].apply(normalize_name)
    info_df['normalized'] = info_df['disease_name'].apply(normalize_name)
    
    # Match
    matched_supp = supplement_df[supplement_df['normalized'] == predicted_clean]
    matched_info = info_df[info_df['normalized'] == predicted_clean]

    # Output
    st.subheader("🧪 Supplement Recommendation")
    if not matched_supp.empty:
        supp = matched_supp.iloc[0]
        st.write(f"**Supplement:** {supp['supplement name']}")
        st.write(f"[Buy Now]({supp['buy link']})")
    else:
        st.write("No supplement info available.")

    st.subheader("📋 Disease Information")
    if not matched_info.empty:
        info = matched_info.iloc[0]
        st.write(f"**Description:** {info['description']}")
        st.write(f"**Steps to Cure:** {info['Possible Steps']}")
        if pd.notnull(info['image_url']):
            st.image(info['image_url'])
    else:
        st.write("No info available.")
