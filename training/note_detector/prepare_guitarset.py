"""
Downloads GuitarSet's annotation files and mono-mic audio recordings into
--output-dir, skipping the download if already present (so CI/Colab caching
works the same way it does for MUSDB18 elsewhere in this project).

GuitarSet: https://guitarset.weebly.com/ — real guitar recordings with
genuine note/string/fret ground truth, openly licensed (CC BY 4.0) for
research use, audio included.

The exact download URLs point at GuitarSet's Zenodo record as of this
writing — check https://guitarset.weebly.com/ for the current link if this
script 404s, since dataset hosting does drift over time.

Usage:
    python3 prepare_guitarset.py --output-dir /path/to/guitarset
"""
import argparse
import os
import subprocess
import zipfile

ZENODO_RECORD = "3371780"
FILES = ["annotation.zip", "audio_mono-mic.zip"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    already_present = os.path.isdir(os.path.join(args.output_dir, "annotation")) and \
        os.path.isdir(os.path.join(args.output_dir, "audio_mono-mic"))
    if already_present:
        print(f"GuitarSet already present at {args.output_dir}, skipping download.")
        return

    for filename in FILES:
        zip_path = os.path.join(args.output_dir, filename)
        url = f"https://zenodo.org/record/{ZENODO_RECORD}/files/{filename}"
        print(f"Downloading {url} ...")
        subprocess.run(["wget", "-q", "--show-progress", "-O", zip_path, url], check=True)

        print(f"Extracting {filename} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(args.output_dir)
        os.remove(zip_path)

    print(f"GuitarSet ready at {args.output_dir}")


if __name__ == "__main__":
    main()
