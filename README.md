## Agro.AI — Crop Disease Detection

Detect crop leaf diseases from images and instantly view actionable guidance and recommended supplements. Agro.AI uses a TensorFlow model with MobileNetV2 feature extraction, wrapped in a simple Flask web app. A Streamlit UI is also included for quick demos.

### ✨ Features
- **Image upload**: Drag-and-drop a leaf image and get a prediction.
- **Fast inference**: MobileNetV2 feature extractor + lightweight classifier.
- **Actionable output**: Disease description, steps to cure, and a buy link for supplements.
- **Two UIs**:
  - Flask + Jinja frontend (production-ready, Dockerized)
  - Streamlit app for quick local demos

### 📁 Project structure
- `app.py`: Flask application (main entry)
- `app1.py`: Streamlit application (optional)
- `templates/index.html`: Frontend template for Flask
- `shuffuled_model.h5`: Trained model weights
- `shuffled_file.csv`: Training feature data and labels
- `disease_info.csv`: Descriptions and steps for diseases
- `supplement_info.csv`: Recommended products and links
- `Dockerfile`: Container image definition (gunicorn entrypoint)
- `requirements.txt`: Python dependencies for Flask app
- `requirement.txt`: Dependencies for Streamlit demo

### 🚀 Quickstart (Local)
1) Create a virtual environment and install dependencies
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2) Run the Flask app
```bash
python app.py
```

3) Open the app
```text
http://127.0.0.1:5000
```

Upload a leaf image (JPG/PNG/JPEG/GIF, ≤ 16 MB) and view the prediction plus recommended actions.

### 🐳 Run with Docker (Recommended)
Build and run the production image using gunicorn.
```bash
docker build -t agro-ai .
docker run -p 5000:5000 agro-ai
```
Open `http://localhost:5000`.

### 🧪 Streamlit demo (Optional)
Install the Streamlit extras and run the demo UI.
```bash
pip install -r requirement.txt
streamlit run app1.py
```
If you are already inside the Flask venv, you can simply `pip install streamlit` and then run the command above.

### 🧠 How it works
1) Image is resized to 224×224 and normalized with MobileNetV2 `preprocess_input`.
2) MobileNetV2 (ImageNet weights, `include_top=False`, `pooling='avg'`) extracts a 1280-d feature vector.
3) Feature vector is standardized using a `StandardScaler` fitted at startup on the training features from `shuffled_file.csv`.
4) TensorFlow model (`shuffuled_model.h5`) predicts the disease class.
5) Predicted label is normalized and matched against `disease_info.csv` and `supplement_info.csv` to show details and a buy link.

### 🧷 File requirements
Place the following files in the project directory (same folder as `app.py`):
- `shuffuled_model.h5`
- `shuffled_file.csv`
- `disease_info.csv`
- `supplement_info.csv`

These are already included in this repository. On first run, if `shuffled_file.csv` is missing, the app attempts to download it via Google Drive (`gdown`).

### ⚙️ Configuration notes
- Max upload size is set to 16 MB.
- Allowed file types: `png`, `jpg`, `jpeg`, `gif`.
- The Flask app disables debug by default. Docker runs with `gunicorn -w 2 -b 0.0.0.0:5000 app:app`.

### 📦 Dependencies
Core (see `requirements.txt`):
- Flask, TensorFlow, Pillow, NumPy, Pandas, scikit-learn, gunicorn, gdown

Streamlit demo (see `requirement.txt`):
- streamlit, TensorFlow, Pandas, scikit-learn, Pillow

### 🔎 Troubleshooting
- "Model file not found": Ensure `shuffuled_model.h5` is in the project directory.
- "CSV file not found": Ensure CSV files exist; the app will attempt to download `shuffled_file.csv` if missing.
- TensorFlow errors in Docker: Prefer building on a stable internet connection; consider pinning TF/Numpy versions if your environment is constrained.
- White/blank page: Check server logs; ensure your image type is among allowed types and file size ≤ 16 MB.

### 📜 License
This repository is provided for educational and research purposes. Add your preferred OSS license if you intend to distribute.

### 🙌 Acknowledgements
- MobileNetV2 by Google (ImageNet pretraining)
- TensorFlow/Keras team
- Dataset metadata curated in `disease_info.csv` and `supplement_info.csv`


