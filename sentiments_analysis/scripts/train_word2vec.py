# scripts/train_word2vec.py
"""
Entraine un Word2Vec avec gensim sur les tokens pré-traités.
Sauvegarde le modèle sur models/word2vec/word2vec.model
"""
from gensim.models import Word2Vec
from pathlib import Path
from utils import ensure_dir, set_seed
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def read_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                yield tokens

def main():
    set_seed(42)
    ensure_dir("models/word2vec")
    train_tokens = Path("data/processed/train_tokens.txt")
    assert train_tokens.exists(), "Exécute preprocess.py avant d'entraîner Word2Vec."

    sentences = list(read_sentences(str(train_tokens)))
    print(f"Nombre de phrases pour entraînement Word2Vec : {len(sentences)}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=200,
        window=5,
        min_count=5,
        workers=4,
        sg=1,         
        epochs=10,
        seed=42
    )

    model.save("models/word2vec/word2vec.model")
    print("Word2Vec entraîné et sauvegardé dans models/word2vec/word2vec.model")

if __name__ == "__main__":
    main()
