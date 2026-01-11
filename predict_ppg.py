import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load ECG data
data = pd.read_csv('ecg_data_20250701_172937.csv')
ecg = data['ECG Amplitude'].values

# Assume PPG is ECG shifted by a certain index for demonstration
ppg = np.roll(ecg, -1)  # Simplified assumption for example

# Create sequences
seq_length = 100
X = []
Y = []
for i in range(len(ecg) - seq_length):
    X.append(ecg[i:i+seq_length])
    Y.append(ppg[i:i+seq_length])
X = np.array(X)
Y = np.array(Y)

# Flatten sequences for Linear Regression
X_flat = X.reshape(X.shape[0], -1)
Y_flat = Y.reshape(Y.shape[0], -1)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_flat, Y_flat, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(Y_test, Y_pred)
print(f'Mean Squared Error: {mse:.4f}')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(Y_test.flatten(), label='Actual PPG', alpha=0.7)
plt.plot(Y_pred.flatten(), label='Predicted PPG', alpha=0.7)
plt.title('Actual vs. Predicted PPG')
plt.xlabel('Sample Index')
plt.ylabel('PPG Amplitude')
plt.legend()
plt.show()
