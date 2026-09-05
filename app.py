import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google import genai

# Sayfa Yapılandırması
st.set_page_config(page_title="CA-AI Engine", page_icon="🤖", layout="wide")

# 1. DİL SEÇİMİ
st.sidebar.title("🌐 Language / Langue / Dil")
lang = st.sidebar.radio("Sélectionnez la langue / Dil Seçin:", ["Français", "Türkçe", "English"])

# Dil Metinleri Sözlüğü
texts = {
    "Français": {
        "welcome": "👋 Bienvenue sur CA-AI !",
        "subtitle": "Votre moteur intelligent complet pour les mathématiques et la physique.",
        "input_label": "Entrez un calcul, une équation ou une fonction :",
        "solve_btn": "Calculer / Résoudre",
        "result": "Résultat :",
        "roots": "Racines (x) :",
        "deriv": "Dérivée :",
        "graph_title": "📊 Graphique de la fonction",
        "photo_tab": "📸 Scanner de Problème (Photo)",
        "photo_label": "Téléchargez la photo de votre problème (Maths / Physique) :",
        "photo_btn": "Analyser et Résoudre",
        "prompt": "Résolvez ce problème de mathématiques/physique étape par étape en français.",
        "api_label": "🔑 Clef d'accès CA-AI Vision :",
        "api_warning": "⚠️ Veuillez entrer votre clef d'accès CA-AI dans le menu latéral."
    },
    "Türkçe": {
        "welcome": "👋 CA-AI'ya Hoş Geldiniz!",
        "subtitle": "Matematik ve fizik için geliştirilmiş kapsamlı akıllı motorunuz.",
        "input_label": "Bir işlem, denklem veya fonksiyon girin:",
        "solve_btn": "Hesapla / Çöz",
        "result": "Sonuç:",
        "roots": "Denklem Kökleri (x):",
        "deriv": "Türevi:",
        "graph_title": "📊 Fonksiyon Grafiği",
        "photo_tab": "📸 Soru Fotoğrafı Tara",
        "photo_label": "Problemin fotoğrafını yükleyin (Matematik / Fizik):",
        "photo_btn": "Fotoğrafı Analiz Et ve Çöz",
        "prompt": "Bu matematik/fizik problemini Türkçe olarak adım adım çöz.",
        "api_label": "🔑 CA-AI Görsel Motor Anahtarı:",
        "api_warning": "⚠️ Lütfen sol menüden CA-AI erişim anahtarınızı girin."
    },
    "English": {
        "welcome": "👋 Welcome to CA-AI!",
        "subtitle": "Your comprehensive intelligent engine for mathematics and physics.",
        "input_label": "Enter a calculation, equation, or function:",
        "solve_btn": "Calculate / Solve",
        "result": "Result:",
        "roots": "Roots (x):",
        "deriv": "Derivative:",
        "graph_title": "📊 Function Graph",
        "photo_tab": "📸 Problem Scanner (Photo)",
        "photo_label": "Upload a photo of your problem (Math / Physics):",
        "photo_btn": "Analyze and Solve",
        "prompt": "Solve this math/physics problem step by step in English.",
        "api_label": "🔑 CA-AI Vision License Key:",
        "api_warning": "⚠️ Please enter your CA-AI access key in the sidebar."
    }
}

t = texts[lang]

# Yan Menü - CA-AI Erişim Anahtarı Girişi
st.sidebar.markdown("---")
api_key = st.sidebar.text_input(t["api_label"], type="password")

# Ana Karşılama Başlığı
st.title(t["welcome"])
st.write(t["subtitle"])
st.markdown("---")

# İşlem Sekmeleri
tab1, tab2 = st.tabs(["🧮 Calcul / Hesaplama Engine", t["photo_tab"]])

# 1. SEKME: HEPSİ BİR ARADA MATEMATİK MOTORU
with tab1:
    girdi = st.text_input(t["input_label"], "2+2")
    
    if st.button(t["solve_btn"]):
        x = sp.symbols('x')
        try:
            expr = sp.sympify(girdi)
            
            if expr.has(x):
                cozum = sp.solve(expr, x)
                turev = sp.diff(expr, x)
                
                st.info(f"**{t['roots']}** {cozum}")
                st.write(f"**{t['deriv']}** {turev}")
                
                st.subheader(t["graph_title"])
                f = sp.lambdify(x, expr, "numpy")
                x_vals = np.linspace(-10, 10, 400)
                y_vals = f(x_vals)
                
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(x_vals, y_vals, color="#1f77b4", linewidth=2)
                ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
                ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
            
            else:
                sonuc = expr.evalf()
                if sonuc % 1 == 0:
                    sonuc = int(sonuc)
                st.success(f"**{t['result']}** {sonuc}")
                
        except Exception as e:
            st.error(f"Erreur / Hata: {e}")

# 2. SEKME: FOTOĞRAF TARAMA VE PROBLEM ÇÖZME
with tab2:
    uploaded_file = st.file_uploader(t["photo_label"], type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Problème / Soru", width=400)
        
        if st.button(t["photo_btn"]):
            if not api_key:
                st.warning(t["api_warning"])
            else:
                try:
                    with st.spinner("CA-AI Analyse... / Tarama yapılıyor..."):
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[img, t["prompt"]]
                        )
                        st.markdown("### 📝 Solution CA-AI / Çözüm:")
                        st.write(response.text)
                except Exception as e:
                    st.error(f"Hata oluştu / Erreur: {e}")
                    

