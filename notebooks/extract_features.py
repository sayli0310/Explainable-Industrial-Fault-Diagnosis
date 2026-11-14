import scipy.io
import numpy as np
import pandas as pd
import os
import glob

from pathlib import Path
from scipy.stats import skew, kurtosis


BASE_DIR = Path(__file__).resolve().parent

# Handle scripts stored inside the notebooks folder
if not (BASE_DIR / "dataset").exists():
    BASE_DIR = BASE_DIR.parent

dataset_path = BASE_DIR / "dataset"
results_path = BASE_DIR / "results"

results_path.mkdir(parents=True, exist_ok=True)

features_list = []

def get_label(filename):

    number = int(filename.split(".")[0])

    if number in [97, 98, 99, 100]:
        return "Normal"

    elif number in [105, 106, 107, 108]:
        return "Inner Race Fault"

    elif number in [118, 119, 120, 121]:
        return "Ball Fault"

    elif number in [130, 131, 132, 133]:
        return "Outer Race Fault"

    else:
        raise ValueError(f"Unknown dataset file: {filename}")
mat_files = sorted(glob.glob(str(dataset_path / "*.mat")))

print("Total MATLAB files found:", len(mat_files))

if len(mat_files) != 16:
    raise ValueError("Expected 16 MATLAB files. Check dataset folder.")
for file in mat_files:

    data = scipy.io.loadmat(file)

    filename = os.path.basename(file)

    # Find Drive End vibration signal
    signal_keys = [
        key for key in data.keys()
        if key.endswith("_DE_time")
    ]

    if not signal_keys:
        raise ValueError(f"Drive End signal not found in {filename}")

    signal = data[signal_keys[0]].flatten()

    segment_size = 2048

    for i in range(0, len(signal) - segment_size + 1, segment_size):

        segment = signal[i:i + segment_size]

        mean = np.mean(segment)

        std = np.std(segment)

        rms = np.sqrt(np.mean(segment ** 2))

        variance = np.var(segment)

        maximum = np.max(segment)

        minimum = np.min(segment)

        peak_to_peak = maximum - minimum

        skewness = skew(segment)

        kurt = kurtosis(segment)

        crest_factor = np.max(np.abs(segment)) / (rms + 1e-10)

        features_list.append({

            "filename": filename,

            "Mean": mean,

            "Std": std,

            "RMS": rms,

            "Variance": variance,

            "Maximum": maximum,

            "Minimum": minimum,

            "Peak_to_Peak": peak_to_peak,

            "Skewness": skewness,

            "Kurtosis": kurt,

            "Crest_Factor": crest_factor,

            "Label": get_label(filename)

        })



df = pd.DataFrame(features_list)


output_file = results_path / "features.csv"

df.to_csv(output_file, index=False)

print("\nFEATURE EXTRACTION COMPLETED!")

print("\nCSV saved at:", output_file)

print("\nDataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nClass Distribution:")
print(df["Label"].value_counts())

print("\nSamples contributed by each file:")

print(
    df.groupby(["filename", "Label"])
      .size()
      .to_string()
)

print("\nUnique files:", df["filename"].nunique())

print("\nMissing values:")
print(df.isnull().sum())