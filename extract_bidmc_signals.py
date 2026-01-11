import wfdb
import numpy as np
import pandas as pd

# Load the bidmc01 record (assumes bidmc01.dat and bidmc01.hea are in the same folder)
record = wfdb.rdrecord('bidmc01')

# Assume channel 0 = ECG, channel 1 = PPG
ecg_signal = record.p_signal[:, 0]
ppg_signal = record.p_signal[:, 1]

# Save as CSV
np.savetxt("bidmc01_ecg.csv", ecg_signal, delimiter=",")
np.savetxt("bidmc01_ppg.csv", ppg_signal, delimiter=",")

print("Saved bidmc01_ecg.csv and bidmc01_ppg.csv")
