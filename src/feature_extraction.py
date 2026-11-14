import scipy.io
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import skew, kurtosis


def extract_features_from_mat(mat_file_path):

    # Convert input into a Path object
    mat_file_path = Path(mat_file_path)

    # Load the uploaded MATLAB file
    data = scipy.io.loadmat(mat_file_path)

    # Find Drive End vibration signal
    signal_keys = [
        key for key in data.keys()
        if key.endswith("_DE_time")
    ]

    if not signal_keys:
        raise ValueError(
            "Drive End vibration signal not found in uploaded file."
        )

    signal = data[signal_keys[0]].flatten()

    segment_size = 2048

    features_list = []

    # Divide signal into non-overlapping segments
    for i in range(
        0,
        len(signal) - segment_size + 1,
        segment_size
    ):

        segment = signal[i:i + segment_size]

        rms = np.sqrt(np.mean(segment ** 2))

        features_list.append({

            "Mean": np.mean(segment),

            "Std": np.std(segment),

            "RMS": rms,

            "Variance": np.var(segment),

            "Maximum": np.max(segment),

            "Minimum": np.min(segment),

            "Peak_to_Peak": np.max(segment) - np.min(segment),

            "Skewness": skew(segment),

            "Kurtosis": kurtosis(segment),

            "Crest_Factor": (
                np.max(np.abs(segment)) / (rms + 1e-10)
            )

        })

    # Convert extracted features into DataFrame
    df = pd.DataFrame(features_list)

    if df.empty:
        raise ValueError(
            "No vibration segments could be extracted from the file."
        )

    return df