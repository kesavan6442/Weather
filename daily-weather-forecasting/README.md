# Daily Weather Forecasting Using Deep Learning (LSTM)

## Project Overview

Daily Weather Forecasting Using Deep Learning is a machine learning project that uses a Long Short-Term Memory (LSTM) neural network to predict daily mean temperature based on historical weather data.

The project uses Python, TensorFlow, Keras, Pandas, NumPy, Matplotlib, and Scikit-learn to preprocess weather data, train the model, evaluate its performance, and visualize temperature predictions.

## Objectives

- Load and preprocess daily weather data.
- Handle missing values and clean the dataset.
- Normalize weather features for deep learning.
- Create time-series sequences from historical weather observations.
- Build and train an LSTM model.
- Evaluate model performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Plot actual versus predicted temperatures.
- Forecast future temperatures using historical weather patterns.

## Project Structure

```text
daily-weather-forecasting/
│
├── data/
│   └── DailyWeatherSynthetic.csv
│
├── src/
│   └── Weather Forecasting.ipynb
│
├── README.md
│
└── requirements.txt
```

### File Description

- **data/DailyWeatherSynthetic.csv** — Contains daily weather data used for training and evaluation.
- **src/Weather Forecasting.ipynb** — Jupyter Notebook containing the data preprocessing, LSTM model implementation, evaluation, visualizations, and forecasting workflow.
- **README.md** — Project documentation.
- **requirements.txt** — Lists the Python libraries required to run the project.

## Technologies Used

- **Python** — Main programming language.
- **Pandas** — Data loading and preprocessing.
- **NumPy** — Numerical calculations and array operations.
- **Matplotlib** — Data visualization and prediction graphs.
- **TensorFlow** — Deep learning framework.
- **Keras** — Building and training the LSTM model.
- **Scikit-learn** — Data normalization and evaluation metrics.
- **Jupyter Notebook** — Interactive development and experimentation.

## Dataset Description

The dataset contains daily weather records with the following features:

| Column | Description |
|---|---|
| `date` | Date of the weather observation |
| `meantemp` | Mean temperature in degrees Celsius |
| `humidity` | Relative humidity |
| `wind_speed` | Wind speed |
| `meanpressure` | Mean atmospheric pressure |

The included dataset contains synthetic demonstration data rather than actual weather-station observations. It is intended for demonstrating the machine learning workflow.

## Data Preprocessing

The preprocessing stage includes:

1. Loading the weather dataset using Pandas.
2. Converting date and numerical columns into appropriate data types.
3. Handling missing or invalid values.
4. Sorting observations chronologically.
5. Normalizing numerical weather features using MinMaxScaler.
6. Creating sequences for LSTM training.

### Input Sequence

The model uses the input shape **(30, 4)**.

- **30** represents the previous 30 days.
- **4** represents four weather features: mean temperature, humidity, wind speed, and mean pressure.

The model uses these historical observations to predict the next day's mean temperature.

## LSTM Model Architecture

The model uses the following architecture:

```text
Historical Weather Dataset
          |
          v
Data Preprocessing
          |
          v
Feature Normalization
          |
          v
Sequence Generation
      Input (30, 4)
          |
          v
    LSTM Layer (64)
          |
          v
     Dropout (0.2)
          |
          v
    LSTM Layer (32)
          |
          v
     Dropout (0.2)
          |
          v
    Dense Layer (16)
       ReLU
          |
          v
    Dense Layer (1)
          |
          v
Predicted Mean Temperature
```

### Model Components

- **LSTM:** Learns patterns from sequential weather observations.
- **Dropout:** Helps reduce overfitting during training.
- **Dense:** Learns relationships between extracted features and the target.
- **Adam optimizer:** Updates model weights during training.
- **Mean Squared Error (MSE):** Used as the training loss function.
- **Early Stopping:** Stops training when validation performance stops improving.

## Model Training

The dataset is divided chronologically into training, validation, and testing sets.

- **Training set:** 70%
- **Validation set:** 15%
- **Testing set:** 15%

The model learns from the training data, uses validation data to monitor performance, and is evaluated on unseen test data.

Chronological splitting helps preserve the time-series order.

## Model Evaluation

The model is evaluated using two metrics.

### 1. Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted temperatures.

A lower MAE indicates smaller average prediction errors.

### 2. Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction errors. It penalizes larger errors more strongly than MAE.

Both metrics are reported in degrees Celsius after converting predictions back to the original temperature scale.

## Visualization

The project can visualize:

- Historical daily mean temperature.
- Training and validation loss.
- Actual versus predicted temperatures on the test dataset.
- Forecast temperatures for future dates.

These visualizations help assess the model's behavior and prediction performance.

## Future Temperature Forecasting

After training, the model can generate temperature predictions for future dates using a sequence of recent weather observations.

A simple recursive forecast can predict the next seven days. If future humidity, wind speed, and pressure are unavailable, keeping them at their latest observed values is only a demonstration assumption and may reduce forecast reliability.

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/daily-weather-forecasting.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Open the Project

```bash
cd daily-weather-forecasting
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Open the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `src/Weather Forecasting.ipynb` and run the cells sequentially.

Alternatively, upload the notebook and dataset to Google Colab, making sure the notebook can access the CSV file.

## Expected Results

After running the notebook, the project should produce:

- A trained LSTM temperature prediction model.
- Test-set MAE and RMSE values.
- An actual-versus-predicted temperature graph.
- Future temperature predictions, if the forecasting section is implemented.

Actual results depend on the dataset and model training process.

## Limitations

- The included dataset is synthetic demonstration data, not real weather observations.
- Performance on synthetic data does not establish real-world forecasting accuracy.
- The future forecast depends on the availability and quality of input weather features.
- The model predicts mean temperature only; it does not directly predict rainfall or other weather conditions.

For real-world evaluation, replace the demonstration dataset with verified historical weather observations and compare the LSTM against a simple forecasting baseline.

## Future Enhancements

- Use real-time weather data from a reliable weather API.
- Compare LSTM with simpler models and other forecasting methods.
- Forecast humidity, wind speed, and atmospheric pressure.
- Develop a web interface for displaying predictions.
- Add location-based forecasting using real historical weather data.

## Conclusion

This project demonstrates how deep learning and LSTM networks can be applied to daily weather time-series forecasting. It covers data preprocessing, sequence generation, model training, performance evaluation, visualization, and future prediction.

The project provides a foundation for further experimentation with real-world weather datasets and more advanced forecasting methods.

## Author

**Kesavan M**

B.E. Computer Science and Engineering
