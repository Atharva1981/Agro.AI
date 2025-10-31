from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import gdown  # Only needed if using Google Drive for large CSVs
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

try:
    # Model is directly in project folder
    model = load_model("shuffuled_model.h5")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Check & download large CSV from Google Drive if not exists
if not os.path.exists("shuffled_file.csv"):
    try:
        csv_url = "https://drive.google.com/uc?id=1_SMwMKvBZwqk_d_tAnRYU-k8vhaJeUJb"
        gdown.download(csv_url, "shuffled_file.csv", quiet=False)
        logger.info("CSV file downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading CSV file: {str(e)}")
        raise

try:
    # Load CSVs
    df = pd.read_csv("shuffled_file.csv")
    supplement_df = pd.read_csv("supplement_info.csv")
    info_df = pd.read_csv("disease_info.csv", encoding='latin-1')
    logger.info("CSV files loaded successfully")
except Exception as e:
    logger.error(f"Error loading CSV files: {str(e)}")
    raise

# Preprocessing setup
le = LabelEncoder()
le.fit(df['label'].values)
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Fit StandardScaler once at startup using training features
if 'image_name' in df.columns:
    _X_train_features = df.drop(columns=['image_name', 'label'])
else:
    _X_train_features = df.drop(columns=['label'])
scaler = StandardScaler()
scaler.fit(_X_train_features)

def normalize_name(name):
    return str(name).strip().lower().replace(" ", "").replace(":", "").replace("|", "").replace("_", "")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    supplement = None
    info = None
    img_url = None
    error = None

    if request.method == "POST":
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                error = "No file uploaded"
                return render_template("index.html", error=error)

            file = request.files['file']
            
            # Check if file is empty
            if file.filename == '':
                error = "No file selected"
                return render_template("index.html", error=error)

            # Check if file type is allowed
            if not allowed_file(file.filename):
                error = f"File type not allowed. Please upload: {', '.join(ALLOWED_EXTENSIONS)}"
                return render_template("index.html", error=error)

            # Process the image
            try:
                img = Image.open(file)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                features = feature_extractor.predict(img_array)[0].reshape(1, -1)
                # Use pre-fitted scaler
                features_scaled = scaler.transform(features)
                features_scaled = np.expand_dims(features_scaled, axis=2)

                pred = model.predict(features_scaled)
                predicted_index = np.argmax(pred)
                predicted_label = le.inverse_transform([predicted_index])[0]

                logger.info(f"Predicted Label (raw): {predicted_label}")

                prediction = predicted_label
                predicted_clean = normalize_name(predicted_label)

                logger.info(f"Predicted Label (normalized): {predicted_clean}")
                supplement_df['normalized'] = supplement_df['disease_name'].apply(normalize_name)
                info_df['normalized'] = info_df['disease_name'].apply(normalize_name)

                logger.info(f"Supplement DF Normalized Names (sample): {supplement_df['normalized'].head().tolist()}")

                matched_supp = supplement_df[supplement_df['normalized'] == predicted_clean]
                matched_info = info_df[info_df['normalized'] == predicted_clean]

                if not matched_supp.empty:
                    supp = matched_supp.iloc[0]
                    supplement = {
                        "name": supp['supplement name'],
                        "link": supp['buy link']
                    }

                if not matched_info.empty:
                    inf = matched_info.iloc[0]
                    info = {
                        "description": inf['description'],
                        "steps": inf['Possible Steps']
                    }
                    if pd.notnull(inf['image_url']):
                        img_url = inf['image_url']

            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                error = "Error processing image. Please try again with a different image."
                return render_template("index.html", error=error)

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            error = "An unexpected error occurred. Please try again."
            return render_template("index.html", error=error)

    return render_template("index.html", prediction=prediction, supplement=supplement, info=info, img_url=img_url, error=error)

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template("index.html", error="File too large. Maximum size is 16MB"), 413

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return render_template("index.html", error="An internal server error occurred. Please try again."), 500

if __name__ == "__main__":
    app.run(debug=False)
