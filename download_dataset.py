import kagglehub
import zipfile
from pathlib import Path

# Define the destination path for the raw data
save_path = Path("data/raw_sroie")
save_path.mkdir(exist_ok=True, parents=True)

print("Downloading SROIE dataset from KaggleHub...")
# Download the dataset zip file to the specified path
try:
    path = kagglehub.dataset_download("ryanznie/sroie-datasetv2-with-labels", path=str(save_path))
    print(f"Dataset downloaded to: {path}")

    # Unzip the downloaded file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(save_path)
        print(f"Dataset extracted to: {save_path}")

    # Clean up the zip file after extraction
    Path(path).unlink()
    print("Cleaned up the zip file.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you have authenticated with Kaggle. Run 'kaggle config set -n username -v [YOUR_USERNAME]' and 'kaggle config set -n key -v [YOUR_KEY]'")

