from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model bundle
model_bundle = joblib.load("model_bundle.pkl")

model_kalori = model_bundle["model_kalori"]
model_protein = model_bundle["model_protein"]
le_kategori = model_bundle["label_encoder_kategori"]
le_kelas = model_bundle["label_encoder_kelas"]

# Gabungkan semua dataset
df_hewani = pd.read_csv("hewani.csv")
df_nabati = pd.read_csv("nabati.csv")
df_buah = pd.read_csv("buahbuahan.csv")

# Tambahkan kolom kategori
df_hewani["kategori"] = "Hewani"
df_nabati["kategori"] = "Nabati"
df_buah["kategori"] = "Buah-buahan"

# Gabungkan semua data
df_data_latih = pd.concat([df_hewani, df_nabati, df_buah], ignore_index=True)

# Hilangkan spasi & standardisasi huruf kecil biar konsisten
df_data_latih["nama_makanan"] = df_data_latih["nama_makanan"].str.strip().str.lower()
df_data_latih["kategori"] = df_data_latih["kategori"].str.strip().str.lower()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    kategori = data.get("kategori", "").strip().lower()
    nama_makanan = data.get("nama_makanan", "").strip().lower()
    berat = data.get("berat")

    # Validasi input kosong
    if not kategori or not nama_makanan or not berat:
        return jsonify({"error": "Semua data harus diisi."})

    # Validasi berat
    try:
        berat = float(berat)
    except ValueError:
        return jsonify({"error": "Berat harus berupa angka."})

    # Validasi kategori
    if kategori not in [k.lower() for k in le_kategori.classes_]:
        return jsonify({"error": f"Kategori '{kategori}' tidak dikenali."})

    # Validasi nama makanan
    if nama_makanan not in [n.lower() for n in le_kelas.classes_]:
        return jsonify({"error": f"Nama makanan '{nama_makanan}' tidak dikenali."})

    # Validasi nama makanan sesuai kategori
    kombinasi_valid = df_data_latih[
        (df_data_latih["kategori"] == kategori) &
        (df_data_latih["nama_makanan"] == nama_makanan)
    ]

    if kombinasi_valid.empty:
        return jsonify({"error": f"Nama makanan '{nama_makanan}' tidak sesuai dengan kategori '{kategori}'."})

    # Encode input (pakai original label encoder)
    # Kita perlu cari index label encoder yang sesuai huruf besar/kecil aslinya
    kategori_original = next(k for k in le_kategori.classes_ if k.lower() == kategori)
    nama_original = next(n for n in le_kelas.classes_ if n.lower() == nama_makanan)

    kategori_encoded = le_kategori.transform([kategori_original])[0]
    nama_encoded = le_kelas.transform([nama_original])[0]

    # Buat DataFrame input sesuai model
    kalori_columns = list(model_kalori.feature_names_in_)
    protein_columns = list(model_protein.feature_names_in_)

    if kalori_columns == protein_columns:
        X_input = pd.DataFrame([{
            kalori_columns[0]: kategori_encoded,
            kalori_columns[1]: nama_encoded,
            kalori_columns[2]: berat
        }])
    else:
        X_input_kalori = pd.DataFrame([{
            kalori_columns[0]: kategori_encoded,
            kalori_columns[1]: nama_encoded,
            kalori_columns[2]: berat
        }])
        X_input_protein = pd.DataFrame([{
            protein_columns[0]: kategori_encoded,
            protein_columns[1]: nama_encoded,
            protein_columns[2]: berat
        }])

    # Prediksi
    if kalori_columns == protein_columns:
        pred_kalori = model_kalori.predict(X_input)[0]
        pred_protein = model_protein.predict(X_input)[0]
    else:
        pred_kalori = model_kalori.predict(X_input_kalori)[0]
        pred_protein = model_protein.predict(X_input_protein)[0]

    return jsonify({
        "kalori": round(pred_kalori, 2),
        "protein": round(pred_protein, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
