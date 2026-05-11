import pandas as pd
import joblib
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset/yogya_house.csv")

print("Kolom dataset:")
print(df.columns)

# Bersihkan harga
def clean_price(price):
    price = str(price)

    if "Miliar" in price:
        number = float(re.findall(r'[\d,]+', price)[0].replace(',', '.'))
        return number * 1_000_000_000

    elif "Juta" in price:
        number = float(re.findall(r'[\d,]+', price)[0].replace(',', '.'))
        return number * 1_000_000

    return None

# Bersihkan ukuran
def clean_area(area):
    return float(str(area).replace(" m²", "").replace(",", "."))

# Cleaning
df['price'] = df['price'].apply(clean_price)
df['surface_area'] = df['surface_area'].apply(clean_area)
df['building_area'] = df['building_area'].apply(clean_area)

# Encode lokasi
le = LabelEncoder()
df['location_encoded'] = le.fit_transform(df['listing-location'])

# Simpan encoder
os.makedirs("model", exist_ok=True)
joblib.dump(le, "model/location_encoder.pkl")

# Hapus data kosong
df = df.dropna()

# Feature
X = df[['surface_area', 'building_area', 'bed', 'bath', 'carport', 'location_encoded']]
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Training
model.fit(X_train, y_train)

# Evaluasi
pred = model.predict(X_test)
score = r2_score(y_test, pred)

print(f"R2 Score: {score:.4f}")

# Save
joblib.dump(model, 'model/random_forest.pkl')

print("Model berhasil disimpan")