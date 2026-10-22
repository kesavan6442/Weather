from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"/"DailyWeatherSynthetic.csv"
MODEL = ROOT/"models"/"weather_lstm.keras"
X_SCALER = ROOT/"models"/"feature_scaler.joblib"
Y_SCALER = ROOT/"models"/"target_scaler.joblib"
OUT = ROOT/"outputs"/"forecast_next_7_days.csv"
FEATURES = ["meantemp", "humidity", "wind_speed", "meanpressure"]
LOOKBACK, DAYS = 30, 7

def main():
    for p in [DATA, MODEL, X_SCALER, Y_SCALER]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run `python src/train.py` first.")
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=FEATURES)
    model = load_model(MODEL)
    xs, ys = joblib.load(X_SCALER), joblib.load(Y_SCALER)
    recent = xs.transform(df[FEATURES].tail(LOOKBACK).to_numpy())
    last_date = df["date"].iloc[-1]
    last_other = df[FEATURES].iloc[-1, 1:].to_numpy(dtype=float)
    rows = []
    for step in range(1, DAYS+1):
        scaled_prediction = model.predict(recent[-LOOKBACK:].reshape(1, LOOKBACK, len(FEATURES)), verbose=0)
        temp = float(ys.inverse_transform(scaled_prediction)[0, 0])
        rows.append({"date": (last_date + pd.Timedelta(days=step)).strftime("%Y-%m-%d"),
                     "predicted_meantemp_C": round(temp, 2)})
        # Demo assumption: other future features remain at their last observed values.
        next_row = xs.transform(np.array([[temp, *last_other]]))
        recent = np.vstack([recent, next_row])
    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved forecast to {OUT}")
    print("Note: future humidity, wind speed and pressure are held at their last observed values.")

if __name__ == "__main__":
    main()
