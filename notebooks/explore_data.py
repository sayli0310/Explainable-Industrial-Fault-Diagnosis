import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load MATLAB file
data = scipy.io.loadmat("dataset/97.mat")

# Extract vibration signal
signal = data["X097_DE_time"].flatten()

# Basic information
print("Signal length:", len(signal))
print("Mean:", np.mean(signal))
print("Standard deviation:", np.std(signal))
print("Minimum:", np.min(signal))
print("Maximum:", np.max(signal))

# Plot vibration signal
plt.figure(figsize=(12, 5))

plt.plot(signal[:2000])

plt.title("Drive End Vibration Signal")
plt.xlabel("Sample Number")
plt.ylabel("Vibration Amplitude")
plt.grid()

plt.show()