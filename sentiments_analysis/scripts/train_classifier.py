# scripts/train_classifier.py
"""
Construit des embeddings de phrase en moyennant les vecteurs Word2Vec,
puis entraine un classifieur (LogisticRegression).
"""
import json
from pathlib import Path
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from utils import ensure_dir
from tqdm import tqdm

def load_processed(split="train"):
    p = Path(f"data/processed/{split}_processed.jsonl")
    assert p.exists()
    texts, labels = [], []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            texts.append(j["text"])
            labels.append(j["label"])
    return texts, labels

def text_to_vec(text, model):
    toks = text.split()
    vecs = []
    for t in toks:
        if t in model.wv:
            vecs.append(model.wv[t])
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)

def main():
    ensure_dir("models/classifiers")
    w2v_path = Path("models/word2vec/word2vec.model")
    assert w2v_path.exists(), "Word2Vec non trouvé, exécute train_word2vec.py d'abord."

    w2v = Word2Vec.load(str(w2v_path))
    X, y = [], []
    texts, labels = load_processed("train")
    print("Conversion des textes en embeddings (train)...")
    for t,l in tqdm(zip(texts, labels), total=len(texts)):
        X.append(text_to_vec(t, w2v))
        y.append(l)
    X = np.vstack(X)
    y = np.array(y)

    # Split pour evaluation locale
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga", n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    print("Validation accuracy:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds))

    joblib.dump(clf, "models/classifiers/logistic_word2vec.joblib")
    print("Classifieur sauvegardé dans models/classifiers/logistic_word2vec.joblib")

if __name__ == "__main__":
    main()
