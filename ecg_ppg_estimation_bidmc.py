import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load ECG and PPG from BIDMC dataset (assumes files like bidmc01.dat/.hea are in the same folder)
record = wfdb.rdrecord('bidmc01', channels=[0, 1])  # Channel 0: ECG, Channel 1: PPG
signals = record.p_signal

# Extract ECG and PPG
ecg_signal = signals[:, 0]
ppg_signal = signals[:, 1]

print("ECG signal shape:", ecg_signal.shape)
print("PPG signal shape:", ppg_signal.shape)

# Use the first 300 samples for training
N = 300
ecg_segment = ecg_signal[:N].reshape(-1, 1)
ppg_segment = ppg_signal[:N].reshape(-1, 1)

# Train a simple linear regression model
model = LinearRegression()
model.fit(ecg_segment, ppg_segment)

# Estimate PPG from ECG
estimated_ppg = model.predict(ecg_segment)

# Plot actual vs estimated PPG
plt.figure(figsize=(10, 5))
plt.plot(ppg_segment, label='Actual PPG')
plt.plot(estimated_ppg, label='Estimated PPG', linestyle='--')
plt.title('PPG Estimation from ECG (First 300 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



