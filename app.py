import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
MODEL_FILENAME = 'KNN_MODEL.pkl' 
SCALER_FILENAME = 'SCALER.pkl'   

def map_activity_scale_to_minutes(scale):
    return int(scale * 10)

def interpret_prediction(score):
    if score >= 9:
        return "Tidur Sangat Baik"
    elif score >= 8:
        return "Tidur Baik"
    elif score >= 7:
        return "Tidur Cukup Baik"
    elif score >= 6:
        return "Tidur Cukup"
    elif score >= 5:
        return "Tidur Kurang"
    else:
        return "Tidur Buruk"

def train_model(data_path='Sleep_health_and_lifestyle_dataset.csv'):
    try:
        df = pd.read_csv(data_path)
        df.dropna(inplace=True)

        features = ['Age', 'Sleep Duration', 'Stress Level', 'Physical Activity Level']
        X = df[features]

        y = df['Quality of Sleep']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) ##80% train 20% test

        model = KNeighborsClassifier(n_neighbors=7)
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_FILENAME)
        joblib.dump(scaler, SCALER_FILENAME)
        print("Model KNN berhasil dilatih dan disimpan.")
        return model, scaler

    except FileNotFoundError:
        print(f"Error: File dataset '{data_path}' tidak ditemukan.")
        return None, None
    except Exception as e:
        print(f"Terjadi error saat melatih model: {e}")
        return None, None

try:
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    print("Model KNN dan Scaler yang sudah ada dimuat.")
except (FileNotFoundError, Exception):
    print("Model atau Scaler tidak ditemukan. Melatih model baru...")
    model, scaler = train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model atau Scaler prediksi belum siap.'}), 500

    try:
        data = request.form
        
        usia = int(data.get('usia'))
        durasi_tidur = float(data.get('durasi_tidur'))
        tingkat_stress = int(data.get('tingkat_stress'))
        aktivitas_skala = int(data.get('aktivitas_fisik_skala'))
        
        aktivitas_fisik_menit = map_activity_scale_to_minutes(aktivitas_skala)
        
        input_data = np.array([[
            usia, 
            durasi_tidur, 
            tingkat_stress, 
            aktivitas_fisik_menit 
        ]])

        input_scaled = scaler.transform(input_data)

        prediction_score = int(model.predict(input_scaled)[0])
        kualitas_tidur_text = interpret_prediction(prediction_score)
        
        return jsonify({
            'skor_prediksi': f'Skor Kualitas: {prediction_score} / 10', 
            'kualitas_tidur': kualitas_tidur_text,
        })

    except (ValueError, TypeError):
        return jsonify({'error': 'Input tidak valid. Pastikan semua terisi.'}), 400
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {e}'}), 500

if __name__ == '__main__':
    if model and scaler:
        print("\n==================================================")
        print("Aplikasi Flask berjalan di http://127.0.0.1:5000/")
        print("==================================================\n")
        app.run(debug=True, use_reloader=False) 
    else:
        print("Gagal menjalankan aplikasi karena model KNN tidak dapat disiapkan.")