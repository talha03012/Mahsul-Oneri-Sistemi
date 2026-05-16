import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🌾 Mahsul Önerici Test")

# Modeli ve Scaler'ı yüklemeye çalışalım
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    st.success("Model ve Scaler başarıyla yüklendi!")
except Exception as e:
    st.error(f"Hata: {e}")

# Basit bir test girişi
n = st.slider("Azot Değeri", 0, 150, 50)

if st.button("Hızlı Test Yap"):
    # Örnek bir veri seti (7 özellik: N, P, K, Temp, Hum, pH, Rain)
    test_data = np.array([[n, 50, 50, 25.0, 80.0, 6.5, 200.0]])
    scaled_data = scaler.transform(test_data)
    tahmin = model.predict(scaled_data)
    st.balloons() # Tebrik balonları