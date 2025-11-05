# scripts/preprocess.py
"""
Prétraitement minimal :
- nettoyage (lower, supprime HTML simple),
- tokenization avec spaCy (fr: si tu veux FR, mais IMDB est en EN),
- sauvegarde des textes nettoyés pour entrainement Word2Vec et Transformers.
"""
import re
from pathlib import Path
import json
from datasets import load_from_disk
import spacy
from tqdm import tqdm
from utils import ensure_dir

# Ajuster le modèle spaCy si nécessaire
SPACY_MODEL = "en_core_web_sm"

def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"<.*?>", " ", text)  
    text = re.sub(r"[^A-Za-z0-9.,!?;:'\"()\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def main():
    raw_dir = Path("data/raw/imdb_dataset")
    assert raw_dir.exists(), "Exécute d'abord download_data.py pour récupérer le dataset."
    ds = load_from_disk(str(raw_dir))

    ensure_dir("data/processed")
    out_dir = Path("data/processed")
    # spaCy charge
    try:
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    except:
        # Toujours exécuter : python -m spacy download en_core_web_sm
        raise RuntimeError(f"spaCy model {SPACY_MODEL} introuvable. Lancez: python -m spacy download {SPACY_MODEL}")

    def tokenize(text):
        doc = nlp(text)
        return [token.lemma_ for token in doc if not token.is_space and not token.is_punct]

    splits = ["train", "test"]
    for split in splits:
        print(f"Processing {split} split")
        texts = ds[split]["text"]
        labels = ds[split]["label"]
        processed = []
        tokens_list = []
        for text,label in tqdm(zip(texts, labels), total=len(texts)):
            c = clean_text(text)
            toks = tokenize(c)
            processed.append({"text": c, "label": int(label)})
            tokens_list.append(toks)

        # sauvegarde JSON lines pour réutilisabilité
        with open(out_dir / f"{split}_processed.jsonl", "w", encoding="utf-8") as f:
            for i, item in enumerate(processed):
                j = {"text": item["text"], "label": item["label"]}
                f.write(json.dumps(j, ensure_ascii=False) + "\n")

        # tokens pour training word2vec
        with open(out_dir / f"{split}_tokens.txt", "w", encoding="utf-8") as f:
            for toks in tokens_list:
                f.write(" ".join(toks) + "\n")

    print("Préprocessing terminé. Fichiers générés dans data/processed/")

if __name__ == "__main__":
    main()
