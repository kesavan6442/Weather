# Daily Weather Forecasting Using Deep Learning (LSTM)

Beginner-friendly time-series project to predict next-day mean temperature from the previous 30 days of weather data.

> **Dataset note:** `data/DailyWeatherSynthetic.csv` contains synthetic demonstration data, not real weather observations. Use verified real-world weather data before making claims about real-world accuracy.

## Project structure
```text
daily-weather-forecasting-lstm/
├── data/DailyWeatherSynthetic.csv
├── src/Weather Forcasting.ipynb
├── models/       # saved model/scalers after training
├── requirements.txt
├── .gitignore
└── README.md
```

## Input shape
`(30, 4)` means 30 previous days and 4 features per day: mean temperature, humidity, wind speed and mean pressure.

## Setup
Python 3.10 or 3.11 is recommended.
```bash
python -m venv .venv
```
Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Install packages:
```bash
pip install -r requirements.txt
```

## Train and evaluate
Run from the project root:
```bash
python src/train.py
```
The script cleans data, checks daily dates, scales data using the training period, creates sequences, trains the LSTM, calculates test MAE/RMSE, and saves the model and results.

## Forecast the next 7 days
After training:
```bash
python src/predict.py
```
The forecast is written to `outputs/forecast_next_7_days.csv`. For this simple demo, future humidity, wind speed and pressure are assumed to stay at their latest observed values. This limits forecast reliability.

## Architecture
```text
Previous 30 days × 4 features
          ↓
      LSTM (64)
          ↓
     Dropout (0.2)
          ↓
      LSTM (32)
          ↓
     Dropout (0.2)
          ↓
   Dense (16, ReLU)
          ↓
       Dense (1)
          ↓
Predicted next-day temperature
```

## Metrics
- **MAE:** average absolute prediction error in °C.
- **RMSE:** square root of average squared error in °C; larger errors are penalized more.
