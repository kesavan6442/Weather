from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "DailyWeatherSynthetic.csv"
MODEL_DIR, OUT_DIR = ROOT / "models", ROOT / "outputs"
FEATURES = ["meantemp", "humidity", "wind_speed", "meanpressure"]
LOOKBACK = 30

def make_sequences(x, y):
    X, Y, indices = [], [], []
    for i in range(LOOKBACK, len(x)):
        X.append(x[i-LOOKBACK:i])
        Y.append(y[i, 0])
        indices.append(i)
    return np.asarray(X), np.asarray(Y), np.asarray(indices)

def main():
    tf.keras.utils.set_random_seed(42)
    MODEL_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"] + FEATURES).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if len(df) < 200:
        raise ValueError("Please provide at least 200 clean daily rows.")
    if not (df["date"].diff().dropna().dt.days == 1).all():
        raise ValueError("Dates must be consecutive daily records.")

    n = len(df)
    train_end, val_end = int(n*0.70), int(n*0.85)
    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    x_scaler.fit(df.loc[:train_end-1, FEATURES])
    y_scaler.fit(df.loc[:train_end-1, ["meantemp"]])
    x = x_scaler.transform(df[FEATURES])
    y = y_scaler.transform(df[["meantemp"]])
    X, Y, idx = make_sequences(x, y)

    tr, va, te = idx < train_end, (idx >= train_end) & (idx < val_end), idx >= val_end
    X_train, Y_train = X[tr], Y[tr]
    X_val, Y_val = X[va], Y[va]
    X_test, Y_test = X[te], Y[te]
    test_idx = idx[te]

    print("Input shape per sample:", (LOOKBACK, len(FEATURES)))
    print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
    model = Sequential([
        Input(shape=(LOOKBACK, len(FEATURES))),
        LSTM(64, return_sequences=True), Dropout(0.2),
        LSTM(32), Dropout(0.2),
        Dense(16, activation="relu"), Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100,
              batch_size=32, shuffle=False,
              callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
              verbose=1)

    pred = model.predict(X_test, verbose=0)
    actual_c = y_scaler.inverse_transform(Y_test.reshape(-1, 1)).ravel()
    pred_c = y_scaler.inverse_transform(pred).ravel()
    mae = float(mean_absolute_error(actual_c, pred_c))
    rmse = float(np.sqrt(mean_squared_error(actual_c, pred_c)))
    metrics = {"test_MAE_C": round(mae, 4), "test_RMSE_C": round(rmse, 4),
               "rows_used": len(df), "lookback_days": LOOKBACK,
               "features": FEATURES, "dataset_note": "Synthetic demonstration data"}
    (OUT_DIR/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    result = pd.DataFrame({"date": df.loc[test_idx, "date"].dt.strftime("%Y-%m-%d").to_numpy(),
                           "actual_meantemp_C": actual_c, "predicted_meantemp_C": pred_c})
    result.to_csv(OUT_DIR/"test_predictions.csv", index=False)
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(result.date), result.actual_meantemp_C, label="Actual")
    plt.plot(pd.to_datetime(result.date), result.predicted_meantemp_C, label="Predicted")
    plt.title("Actual vs Predicted Daily Mean Temperature")
    plt.xlabel("Date"); plt.ylabel("Temperature (°C)"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR/"actual_vs_predicted.png", dpi=150); plt.close()
    model.save(MODEL_DIR/"weather_lstm.keras")
    joblib.dump(x_scaler, MODEL_DIR/"feature_scaler.joblib")
    joblib.dump(y_scaler, MODEL_DIR/"target_scaler.joblib")
    print(f"Test MAE: {mae:.4f} °C")
    print(f"Test RMSE: {rmse:.4f} °C")
    print("Saved model, metrics, plot and predictions.")

if __name__ == "__main__":
    main()
