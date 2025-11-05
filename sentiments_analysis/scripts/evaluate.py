# scripts/evaluate.py
"""
Compare Word2Vec+Logistic et DistilBERT finetuné sur le set test.
Affiche accuracy, F1, matrice de confusion pour chaque modèle.
"""
import json
from pathlib import Path
import numpy as np
from gensim.models import Word2Vec
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from utils import ensure_dir

def load_test_texts():
    p = Path("data/processed/test_processed.jsonl")
    assert p.exists(), "Fichier test_processed.jsonl manquant. Exécute preprocess.py"
    texts, labels = [], []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            texts.append(j["text"])
            labels.append(int(j["label"]))
    return texts, labels

def text_to_vec_mean(text, w2v):
    toks = text.split()
    vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
    if not vecs:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)

def eval_word2vec():
    w2v = Word2Vec.load("models/word2vec/word2vec.model")
    clf = joblib.load("models/classifiers/logistic_word2vec.joblib")
    texts, labels = load_test_texts()
    X = np.vstack([text_to_vec_mean(t, w2v) for t in texts])
    preds = clf.predict(X)
    print("=== Word2Vec + LogisticRegression ===")
    print("Accuracy:", accuracy_score(labels, preds))
    print(classification_report(labels, preds))
    print("Confusion matrix:\n", confusion_matrix(labels, preds))

def eval_distilbert():
    # we use transformers pipeline for simplicity
    print("Chargement du modèle DistilBERT finetuné...")
    classifier = pipeline("text-classification", model="models/distilbert_finetuned_final", device=0 if __import__("torch").cuda.is_available() else -1)
    texts, labels = load_test_texts()
    batch_size = 64
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        outs = classifier(batch, truncation=True, padding=True)
        preds.extend([1 if o["label"].lower().endswith("positive") or o["label"]=="LABEL_1" or o["label"]=="POSITIVE" else 0 for o in outs])
    print("=== DistilBERT finetuné ===")
    print("Accuracy:", accuracy_score(labels, preds))
    print(classification_report(labels, preds))
    from sklearn.metrics import confusion_matrix
    print("Confusion matrix:\n", confusion_matrix(labels, preds))

def main():
    eval_word2vec()
    eval_distilbert()

if __name__ == "__main__":
    main()
