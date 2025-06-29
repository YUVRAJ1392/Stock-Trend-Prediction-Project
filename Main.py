# Imports
from tensorflow import keras 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from datetime import datetime
from sklearn.metrics import mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



data = pd.read_csv("TAMO Historical Data.csv")
print(data.head())
print(data.info())
print(data.describe())


# Initial Data Visualization
# Plot 1 - Open and Close Prices of time
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Open'], label="Open",color="blue")
plt.plot(data['Date'], data['Price'], label="Close",color="red")
plt.title("Open-Close Price over Time")
plt.legend()
# plt.show()

# Plot 2 - Trading Volume (check for outliers)
plt.figure(figsize=(12,6))
plt.plot(data['Date'],data['Vol.'],label="Volume",color="orange")
plt.title("Stock Volume over Time")
# plt.show()

# Drop non-numeric columns
numeric_data = data.select_dtypes(include=["int64","float64"])

# Plot 3 - Check for correlation between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
# plt.show()

# Convert the Data into Date time then create a date filter
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

prediction = data.loc[
    (data['Date'] > datetime(2020,1,1)) &
    (data['Date'] < datetime(2024,1,1))
]

plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Price'],color="blue")
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price over time")


# Prepare for the LSTM Model (Sequential)
stock_close = data.filter(["Price"])
dataset = stock_close.values #convert to numpy array
training_data_len = int(np.ceil(len(dataset) * 0.67))

# Preprocessing Stages
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len] #67% of all out data

X_train, y_train = [], []


# Create a sliding window for our stock (60 days)
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Build the Model
model = keras.models.Sequential()

# First Layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],1)))

# Second Layer
model.add(keras.layers.LSTM(64, return_sequences=False))

# 3rd Layer (Dense)
model.add(keras.layers.Dense(128, activation="relu"))

# 4th Layer (Dropout)
model.add(keras.layers.Dropout(0.5))

# Final Output Layer
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer="adam",
              loss="mae",
              metrics=[keras.metrics.RootMeanSquaredError()])


training = model.fit(X_train, y_train, epochs=100, batch_size=32)


# Prep the test data
test_data = scaled_data[training_data_len - 60:]
X_test, y_test = [], dataset[training_data_len:]


for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1 ))


# Make a Prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# Plotting data
train = data[:training_data_len]
test =  data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12,8))
plt.plot(train['Date'], train['Price'], label="Train (Actual)", color='blue')
plt.plot(test['Date'], test['Price'], label="Test (Actual)", color='orange')
plt.plot(test['Date'], test['Predictions'], label="Predictions", color='red')
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
# plt.show()

# Predict Next 10 Days
x_input = test_data[-90:].reshape(1, -1)
temp_input = list(x_input[0])
lst_output = []

for i in range(10):
    if len(temp_input) > 90:
        x_input = np.array(temp_input[-90:]).reshape(1, 90, 1)
    else:
        x_input = np.array(temp_input).reshape(1, 90, 1)
    yhat = model.predict(x_input, verbose=0)
    lst_output.extend(yhat.tolist())
    temp_input.append(yhat[0][0])

rmse = np.sqrt(mean_squared_error(test['Price'], test['Predictions']))
print(f"Test RMSE: {rmse}")

# Plot Future Predictions
day_new = np.arange(1, 91)
day_pred = np.arange(91, 101)

plt.figure(figsize=(12, 6))
plt.plot(day_new, scaler.inverse_transform(scaled_data[-90:]), label="Past 90 Days")
plt.plot(day_pred, scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)), label="Next 10 Days")
plt.title("Next 10-Day Forecast")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()