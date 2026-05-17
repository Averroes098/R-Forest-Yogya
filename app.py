from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import json

app = Flask(__name__)

# Load model dan encoder
model = joblib.load("model/model_rumah_jogja_rf_tuned.pkl")
le_district = joblib.load("model/encoder_district.pkl")
le_city = joblib.load("model/encoder_city.pkl")
le_furnishing = joblib.load("model/encoder_furnishing.pkl")

# Load mapping kota -> kecamatan
with open('city_district_mapping.json', 'r') as f:
    city_district_map = json.load(f)

# Filter nilai 'Unknown' dari kota dan furnishing
city_choices = [c for c in le_city.classes_.tolist() if c != 'Unknown']
furnishing_choices = [f for f in le_furnishing.classes_.tolist() if f != 'Unknown']

# Mapping furnishing ke Bahasa Indonesia untuk ditampilkan
furnishing_display_map = {
    'Furnished': 'Berperabot',
    'Semi Furnished': 'Semi Berperabot',
    'Unfurnished': 'Tidak Berperabot'
}
# Jika ada 'Unknown' tidak ditampilkan, jadi abaikan

# Urutkan
city_choices.sort()
furnishing_choices.sort()

# Buat daftar pilihan furnishing dengan (value, label)
furnishing_options = [(f, furnishing_display_map.get(f, f)) for f in furnishing_choices]

features = [
    'landSize', 'buildingSize', 'bedrooms', 'bathrooms', 'floors', 'garages',
    'electricity', 'total_rooms', 'land_building_ratio',
    'district_enc', 'city_enc', 'furnishing_enc'
]

@app.route('/')
def index():
    return render_template('index.html',
                           cities=city_choices,
                           furnishing_options=furnishing_options)

@app.route('/get_districts/<city>')
def get_districts(city):
    districts = city_district_map.get(city, [])
    districts = [d for d in districts if d != 'Unknown']
    return jsonify(sorted(districts))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        landSize = float(request.form['landSize'])
        buildingSize = float(request.form['buildingSize'])
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        floors = float(request.form['floors'])
        garages = float(request.form['garages'])
        electricity = float(request.form['electricity'])
        city = request.form['city']
        district = request.form['district']
        furnishing = request.form['furnishing']  # value asli (Furnished, Semi Furnished, Unfurnished)

        total_rooms = bedrooms + bathrooms
        land_building_ratio = landSize / (buildingSize + 1e-6)

        district_enc = le_district.transform([district])[0] if district in le_district.classes_ else -1
        city_enc = le_city.transform([city])[0] if city in le_city.classes_ else -1
        furnishing_enc = le_furnishing.transform([furnishing])[0] if furnishing in le_furnishing.classes_ else -1

        input_data = pd.DataFrame([[
            landSize, buildingSize, bedrooms, bathrooms, floors, garages,
            electricity, total_rooms, land_building_ratio,
            district_enc, city_enc, furnishing_enc
        ]], columns=features)

        prediction = model.predict(input_data)[0]
        price_formatted = f"Rp{prediction:,.0f}".replace(",", ".")

        return render_template('index.html',
                               prediction=price_formatted,
                               cities=city_choices,
                               city_selected=city,
                               furnishing_options=furnishing_options)

    except Exception as e:
        return render_template('index.html',
                               error=str(e),
                               cities=city_choices,
                               furnishing_options=furnishing_options)

if __name__ == '__main__':
    app.run(debug=True)