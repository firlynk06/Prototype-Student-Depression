import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# ----------------------------
# KONFIGURASI HALAMAN
# ----------------------------
st.set_page_config(
    page_title="Skrining Awal Depresi Mahasiswa",
    page_icon="🧠",
    layout="centered"
)

SLEEP_CATEGORIES = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours', 'Others']
DIET_CATEGORIES = ['Unhealthy', 'Moderate', 'Healthy', 'Others']

# ----------------------------
# LOAD & TRAIN (cached, hanya jalan sekali)
# ----------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("Student_Depression_Dataset.csv")

    # Samakan urutan langkah dengan notebook asli
    df.drop(columns=['Have you ever had suicidal thoughts ?'], inplace=True)
    df.dropna(subset=['Financial Stress'], inplace=True)

    cols_to_drop = ['id', 'Gender', 'Profession', 'Work Pressure', 'Job Satisfaction', 'City']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True)

    # Label Encoding: Sleep Duration & Dietary Habits
    df['Sleep Duration'] = df['Sleep Duration'].astype(str)
    df['Dietary Habits'] = df['Dietary Habits'].astype(str)
    df.loc[~df['Sleep Duration'].isin(SLEEP_CATEGORIES), 'Sleep Duration'] = 'Others'
    df.loc[~df['Dietary Habits'].isin(DIET_CATEGORIES), 'Dietary Habits'] = 'Others'

    le_sleep = LabelEncoder().fit(SLEEP_CATEGORIES)
    le_diet = LabelEncoder().fit(DIET_CATEGORIES)
    df['Sleep Duration'] = le_sleep.transform(df['Sleep Duration'])
    df['Dietary Habits'] = le_diet.transform(df['Dietary Habits'])

    # One-Hot Encoding: Degree, Family History, City
    enc_degree = OneHotEncoder(handle_unknown='ignore')
    ohe_degree = enc_degree.fit_transform(df[['Degree']])
    degree_cols = [f"Degree_{c}" for c in enc_degree.categories_[0]]
    ohe_degree_df = pd.DataFrame(ohe_degree.toarray(), columns=degree_cols, index=df.index)

    enc_family = OneHotEncoder(handle_unknown='ignore')
    ohe_family = enc_family.fit_transform(df[['Family History of Mental Illness']])
    family_cols = [f"Family_{c}" for c in enc_family.categories_[0]]
    ohe_family_df = pd.DataFrame(ohe_family.toarray(), columns=family_cols, index=df.index)

    df_final = pd.concat(
        [df.drop(columns=['Degree', 'Family History of Mental Illness']),
         ohe_degree_df, ohe_family_df],
        axis=1
    )

    X = df_final.drop(columns=['Depression'])
    y = df_final['Depression']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = svm.SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    clf.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, clf.predict(X_train))
    acc_test = accuracy_score(y_test, clf.predict(X_test))

    # Simpan daftar kategori asli (untuk dropdown UI) & urutan kolom X (untuk prediksi)
    meta = {
        "degree_categories": list(enc_degree.categories_[0]),
        "family_categories": list(enc_family.categories_[0]),
        "feature_columns": list(X.columns),
        "acc_train": acc_train,
        "acc_test": acc_test,
    }

    objs = {
        "le_sleep": le_sleep,
        "le_diet": le_diet,
        "enc_degree": enc_degree,
        "enc_family": enc_family,
        "scaler": scaler,
        "clf": clf,
    }

    return objs, meta


objs, meta = load_and_train()

# ----------------------------
# HEADER
# ----------------------------
st.title("🧠 Skrining Awal Depresi Mahasiswa")
st.caption(
    "Prototipe alat bantu skrining awal berbasis Machine Learning (SVM - RBF Kernel). "
    "Bukan alat diagnosis medis — hasil hanya indikasi awal, bukan pengganti konsultasi profesional."
)

with st.expander("ℹ️ Tentang model ini"):
    st.write(
        f"""
        Model dilatih menggunakan **Student Depression Dataset** dengan algoritma **SVM kernel RBF**.
        - Akurasi data training: **{meta['acc_train']*100:.2f}%**
        - Akurasi data testing: **{meta['acc_test']*100:.2f}%**

        Fitur yang digunakan: usia, tekanan akademik, IPK, kepuasan belajar, durasi tidur,
        pola makan, jenjang pendidikan, jam belajar/kerja, tekanan finansial, dan riwayat keluarga
        dengan gangguan mental.

        Fitur kota domisili sengaja tidak digunakan meskipun signifikan secara statistik pada
        dataset asal (India), karena kategorinya tidak relevan/tidak dapat digeneralisasi untuk
        konteks pengguna di Indonesia.

        Input IPK ditampilkan dalam skala 0-4 (skala umum di Indonesia) dan dikonversi secara
        proporsional ke skala 0-10 (skala dataset asli) sebelum diproses oleh model.
        """
    )

st.divider()

# ----------------------------
# FORM INPUT
# ----------------------------
st.subheader("Isi data berikut")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", min_value=15, max_value=60, value=20)
    academic_pressure = st.slider("Tekanan Akademik (0 = rendah, 5 = tinggi)", 0, 5, 3)
    ipk = st.number_input("IPK (skala 0-4)", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    cgpa = ipk * 2.5  # konversi ke skala 0-10 (skala dataset asli/India) untuk keperluan model
    study_satisfaction = st.slider("Kepuasan Belajar (0 = rendah, 5 = tinggi)", 0, 5, 3)
    work_study_hours = st.number_input("Jam Belajar/Kerja per Hari", min_value=0, max_value=24, value=6)

with col2:
    financial_stress = st.slider("Tekanan Finansial (1 = rendah, 5 = tinggi)", 1, 5, 2)
    sleep_duration = st.selectbox("Durasi Tidur", SLEEP_CATEGORIES)
    dietary_habits = st.selectbox("Pola Makan", DIET_CATEGORIES)
    degree = st.selectbox("Jenjang / Program Studi", sorted(meta["degree_categories"]))
    family_history = st.selectbox("Riwayat Keluarga dengan Gangguan Mental", sorted(meta["family_categories"]))

st.divider()

# ----------------------------
# PREDIKSI
# ----------------------------
if st.button("🔍 Cek Hasil Skrining", use_container_width=True):
    # Susun 1 baris input sesuai skema training
    row = {
        "Age": age,
        "Academic Pressure": academic_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Sleep Duration": objs["le_sleep"].transform([sleep_duration])[0],
        "Dietary Habits": objs["le_diet"].transform([dietary_habits])[0],
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
    }
    base_df = pd.DataFrame([row])

    degree_vec = objs["enc_degree"].transform([[degree]]).toarray()
    degree_df = pd.DataFrame(degree_vec, columns=[f"Degree_{c}" for c in objs["enc_degree"].categories_[0]])

    family_vec = objs["enc_family"].transform([[family_history]]).toarray()
    family_df = pd.DataFrame(family_vec, columns=[f"Family_{c}" for c in objs["enc_family"].categories_[0]])

    full_row = pd.concat([base_df, degree_df, family_df], axis=1)
    full_row = full_row.reindex(columns=meta["feature_columns"], fill_value=0)

    X_input = objs["scaler"].transform(full_row)
    pred = objs["clf"].predict(X_input)[0]
    proba = objs["clf"].predict_proba(X_input)[0]

    st.subheader("Hasil")
    if pred == 1:
        st.error(f"Indikasi kecenderungan **depresi** (probabilitas {proba[1]*100:.1f}%).")
        st.write(
            "Ini bukan diagnosis. Jika kamu merasa berat menjalani hari-hari belakangan ini, "
            "pertimbangkan untuk berbicara dengan konselor kampus, psikolog, atau orang yang kamu percaya."
        )
    else:
        st.success(f"Indikasi kecenderungan **tidak depresi** (probabilitas {proba[0]*100:.1f}%).")
        st.write("Tetap jaga kesehatan mental dengan istirahat cukup dan keseimbangan aktivitas.")

    st.caption("Catatan: hasil ini adalah prediksi statistik dari model, bukan penilaian klinis.")

st.divider()
st.caption("Prototipe skripsi/proyek akademik — Sains Data, dikembangkan untuk keperluan portofolio & artikel beasiswa.")
