from PIL import Image

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ultralytics import YOLO

from utils.model import choquet_integral, load_model, load_vgg16
from utils.image import extract_features_from_uploaded_file
from utils.yolo import get_ratio_data

# ======================
# Load Models
# ======================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

vgg16_model = load_vgg16()
rf_model, svr_model, xgb_model, lgbm_model = load_model()

yolo_back_model = YOLO(os.path.join(MODEL_DIR, 'yolo-belakang.pt'))
yolo_side_model = YOLO(os.path.join(MODEL_DIR, 'yolo-samping.pt'))

scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

choquet_weights = np.array([0.21676177, 0.00038352, 0.07302243, 0.10925336])

# ======================
# UI
# ======================
st.title("Estimasi Produksi Susu Sapi")
st.subheader("Algoritma Machine Learning - Fuzzy Integral")
st.write("Unggah gambar sapi untuk mengestimasi jumlah susu (liter).")

# ======================
# File Upload
# ======================
col1, col2 = st.columns(2)
with col1:
    side_image = st.file_uploader(
        "📸 Gambar Samping",
        type=["jpg", "jpeg", "png"],
        key="side_img"
    )
with col2:
    back_image = st.file_uploader(
        "📸 Gambar Belakang",
        type=["jpg", "jpeg", "png"],
        key="back_img"
    )

# ======================
# Preview Images
# ======================
preview_col1, preview_col2 = st.columns(2)
with preview_col1:
    if side_image:
        st.markdown("**Preview Gambar Samping**")
        st.image(Image.open(side_image), width=200)

with preview_col2:
    if back_image:
        st.markdown("**Preview Gambar Belakang**")
        st.image(Image.open(back_image), width=200)

# ======================
# Prediction
# ======================
if st.button("Estimasi Susu (Liter)"):
    if side_image is not None and back_image is not None:

        # Extract image features
        side_features = extract_features_from_uploaded_file(
            side_image, vgg16_model
        )
        back_features = extract_features_from_uploaded_file(
            back_image, vgg16_model
        )
        
        side_ratio_features = get_ratio_data(
            side_image, yolo_side_model, 'side'
        )
        back_ratio_features = get_ratio_data(
            back_image, yolo_back_model, 'back'
        )
        
        side_features_np = np.asarray(side_features).reshape(1, -1)
        back_features_np = np.asarray(back_features).reshape(1, -1)
        
        side_ratio_np = np.asarray(side_ratio_features).reshape(1, -1)
        back_ratio_np = np.asarray(back_ratio_features).reshape(1, -1)
        
        X = np.hstack([
            back_features_np,
            side_features_np,
            back_ratio_np,
            side_ratio_np
        ])

        X_scaled = scaler.transform(X)

        # # Convert to DataFrame
        # X_side_df = pd.DataFrame([side_features])
        # X_back_df = pd.DataFrame([back_features])

        # X_side_df.columns = [f"{i+1}s" for i in range(X_side_df.shape[1])]
        # X_back_df.columns = [f"{i+1}b" for i in range(X_back_df.shape[1])]

        # # Combine only image features
        # X_combined_df = pd.concat(
        #     [X_side_df, X_back_df],
        #     axis=1
        # )

        # X = X_combined_df.values

        # Model predictions
        rf_pred = rf_model.predict(X_scaled)
        svr_pred = svr_model.predict(X_scaled)
        xgb_pred = xgb_model.predict(X_scaled)
        lgbm_pred = lgbm_model.predict(X_scaled)

        model_prediction = np.array([
            rf_pred[0],
            svr_pred[0],
            xgb_pred[0],
            lgbm_pred[0]
        ])

        choquet_result = choquet_integral(
            choquet_weights,
            model_prediction
        )

        # ======================
        # Output
        # ======================
        st.subheader("📊 Hasil Estimasi:")
        st.write(f"🔸 Random Forest: **{rf_pred[0]:.2f} liter**")
        st.write(f"🔸 Support Vector Regression: **{svr_pred[0]:.2f} liter**")
        st.write(f"🔸 XGBoost: **{xgb_pred[0]:.2f} liter**")
        st.write(f"🔸 LightGBM: **{lgbm_pred[0]:.2f} liter**")
        st.success(
            f"⭐ Estimasi Akhir (Choquet): **{choquet_result:.2f} liter**"
        )

    else:
        st.warning(
            "Mohon unggah kedua gambar: Gambar Samping dan Gambar Belakang."
        )

# ======================
# Logo (Bottom Right)
# ======================
st.markdown(
    """
    <style>
    .bottom-right-image {
        position: fixed;
        bottom: 20px;
        right: 20px;
        opacity: 0.9;
        z-index: 100;
    }
    </style>
    <div class="bottom-right-image">
        <img src="https://www.deheus.id/contentassets/f5926d4cb9f74e2a91d2545bdc517e48/imagefk73.png" width="240">
    </div>
    """,
    unsafe_allow_html=True
)
