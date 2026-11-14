import os
from pathlib import Path

folder = Path("dataset")

files = sorted(folder.glob("*.mat"))

print("Total MATLAB files:", len(files))
print("-" * 45)

for file in files:
    size = file.stat().st_size

    with open(file, "rb") as f:
        header = f.read(20)

    valid = header.startswith(b"MATLAB 5.0 MAT-file")

    print(f"{file.name:12} | {size:10} bytes | "
          f"{'VALID' if valid else 'INVALID'}")

print("-" * 45)

if len(files) == 16 and all(
    f.open("rb").read(20).startswith(b"MATLAB 5.0 MAT-file")
    for f in files
):
    print("SUCCESS: All 16 MATLAB files verified!")
else:
    print("Some files need attention.")