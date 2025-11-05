# scripts/download_data.py
"""
Télécharge et sauvegarde le dataset IMDB via Hugging Face datasets.
Sauvegarde sous data/raw/imdb_dataset.arrow pour réutilisation rapide.
"""
from datasets import load_dataset
from pathlib import Path

def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Téléchargement du dataset IMDB (huggingface)...")
    ds = load_dataset("imdb")

    # Sauvegarder en format Hugging Face dataset 
    print("Sauvegarde locale du dataset (format dataset.save_to_disk)...")
    ds.save_to_disk(str(out_dir / "imdb_dataset"))

    print("Terminé. Dataset sauvegardé dans data/raw/imdb_dataset")

if __name__ == "__main__":
    main()
