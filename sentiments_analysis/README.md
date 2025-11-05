# 🎬 Analyse de Sentiment - NLP & Transformers

## 🧠 Objectif
Ce projet vise à construire un modèle capable de déterminer si une critique de film est **positive** ou **négative**.  
Il illustre la transition entre les **représentations classiques du langage (Word2Vec)** et les **modèles modernes basés sur les Transformers (DistilBERT)**.

---

## 🪜 Étapes du projet
1. **Prétraitement & exploration** des données (IMDB dataset)
2. **Entraînement Word2Vec** + classification traditionnelle
3. **Fine-tuning DistilBERT** sur les mêmes données
4. **Comparaison des performances**
5. **Interface Streamlit** pour tester le modèle

---

## 📂 Structure
sentiment_analysis/
├── data/ # Données brutes et traitées
├── models/ # Modèles enregistrés
├── notebooks/ # Notebooks d'expérimentation
├── streamlit_app.py # Interface de démonstration
├── requirements.txt
└── README.md

sentiment_analysis/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── word2vec/
│   │   └── word2vec.model
│   ├── classifiers/
│   │   └── logistic_word2vec.joblib
│   └── distilbert_finetuned/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_word2vec_training.ipynb
│   ├── 03_transformers_finetuning.ipynb
│   └── 04_evaluation_and_comparison.ipynb
├── scripts/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train_word2vec.py
│   ├── train_classifier.py
│   ├── finetune_transformer.py
│   └── evaluate.py
├── streamlit_app.py
├── utils.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation
```bash
git clone https://github.com/<votre_username>/sentiment_analysis.git
cd sentiment_analysis
pip install -r requirements.txt


Pour lancer : streamlit run streamlit_app.py


🧩 Technologies

Python, Pandas, NumPy, Scikit-learn

NLTK / spaCy / Gensim

Transformers (Hugging Face), PyTorch

Streamlit pour le déploiement

✨ Résultat attendu

Une application simple et interactive :

Entrée texte → modèle DistilBERT → résultat de sentiment

Comparaison entre Word2Vec et Transformers (via notebooks)



# Analyse de Sentiment - IMDB (Word2Vec vs DistilBERT)

## Installation
1. Crée un virtualenv / conda env
2. `pip install -r requirements.txt`
3. `python -m spacy download en_core_web_sm`

## Pipeline recommandé
1. `python scripts/download_data.py`
2. `python scripts/preprocess.py`
3. `python scripts/train_word2vec.py`
4. `python scripts/train_classifier.py`
5. `python scripts/finetune_transformer.py`
6. `python scripts/evaluate.py`
7. `streamlit run streamlit_app.py`

## Structure
(voir la section Structure plus haut)

## Notes
- DistilBERT fine-tuning : si tu as GPU, Trainer utilisera CUDA; sinon utilisation CPU (moins rapide).
- Pour HP/Production : utiliser des mécanismes de logging, checkpoints plus fréquents, scheduler LR, et validation holdout.

