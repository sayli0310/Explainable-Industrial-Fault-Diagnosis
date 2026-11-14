import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = scipy.io.loadmat("dataset/100.mat")

# Extract vibration signals
DE = data["X100_DE_time"].flatten()
FE = data["X100_FE_time"].flatten()

# Create time axis
time_DE = np.arange(len(DE))
time_FE = np.arange(len(FE))

# Plot signals
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(time_DE[:2000], DE[:2000])
plt.title("Drive End Vibration Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time_FE[:2000], FE[:2000])
plt.title("Fan End Vibration Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

# Statistical comparison
print("DRIVE END STATISTICS")
print("Mean:", np.mean(DE))
print("Standard Deviation:", np.std(DE))
print("Maximum:", np.max(DE))
print("Minimum:", np.min(DE))

print("\nFAN END STATISTICS")
print("Mean:", np.mean(FE))
print("Standard Deviation:", np.std(FE))
print("Maximum:", np.max(FE))
print("Minimum:", np.min(FE))