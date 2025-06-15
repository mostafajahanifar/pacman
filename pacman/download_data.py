import os
import sys

import requests

from pacman.config import DATA_DIR

print(7 * "=" * 7)
print("Downloading data necessary for experiments from Zenodo")
print(7 * "=" * 7)

ZENODO_RECORD_ID = "15603656"  # check this one
API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def list_files(token=None):
    params = {}
    if token:
        params["access_token"] = token
    resp = requests.get(API_URL, params=params)
    resp.raise_for_status()
    files = resp.json().get("files", [])
    return [(f["key"], f["links"]["self"]) for f in files]


def download_file(url, target_dir, filename, token=None):
    params = {"access_token": token} if token else {}
    resp = requests.get(url, params=params, stream=True)
    resp.raise_for_status()
    path = os.path.join(target_dir, filename)
    with open(path, "wb") as fp:
        for chunk in resp.iter_content(chunk_size=8192):
            fp.write(chunk)
    print(f"✅ Saved: {path}")


def main():
    token = os.getenv("ZENODO_TOKEN", None)
    print(f"Listing files in record {ZENODO_RECORD_ID}...")
    files = list_files(token)
    if not files:
        print("No files found or this record is restricted.")
        sys.exit(1)

    print("Available files:")
    for i, (name, _) in enumerate(files, 1):
        print(f"{i:>2}. {name}")

    choice = input(
        "\nEnter file numbers to download (e.g. 1,3) or 'all' to download everything: "
    ).strip()
    if choice.lower() == "all":
        picks = range(1, len(files) + 1)
    else:
        try:
            picks = [int(x) for x in choice.split(",") if x.strip().isdigit()]
        except ValueError:
            print("Invalid input. Exiting.")
            sys.exit(1)

    for i in picks:
        if i < 1 or i > len(files):
            print(f"❌ Invalid file number: {i}")
            continue
        name, url = files[i - 1]
        print(f"Downloading {name}...")
        download_file(url, DATA_DIR, name, token)

    print("\nAll done!")


if __name__ == "__main__":
    main()
