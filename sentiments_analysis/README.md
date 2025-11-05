# 🎬 Analyse de Sentiment - NLP & Transformers

## Objectif
Ce projet vise à construire un modèle capable de déterminer si une critique de film est **positive** ou **négative**.  
Il illustre la transition entre les **représentations classiques du langage (Word2Vec)** et les **modèles modernes basés sur les Transformers (DistilBERT)**.

---

## Étapes du projet
1. **Prétraitement & exploration** des données (IMDB dataset)
2. **Entraînement Word2Vec** + classification traditionnelle
3. **Fine-tuning DistilBERT** sur les mêmes données
4. **Comparaison des performances**
5. **Interface Streamlit** pour tester le modèle

---

## Structure
sentiment_analysis/ \\
│
├── scripts/                # Scripts d'entraînement et d'évaluation
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train_word2vec.py
│   ├── train_classifier.py
│   ├── finetune_transformer.py
│   └── evaluate.py
│
├── models/                 # Modèles enregistrés (Word2Vec, DistilBERT)
├── data/                   # Données brutes et prétraitées
├── notebooks/              # Expérimentations et visualisations
├── utils.py                # Fonctions utilitaires
├── streamlit_app.py        # Interface utilisateur
├── requirements.txt
└── README.md


---

## Installation

#### Cloner le dépôt
git clone https://github.com/28-KONE/portfolio-data-science.git
cd sentiment_analysis

#### Installer les dépendances
pip install -r requirements.txt

#### Télécharger le modèle spaCy 
python -m spacy download en_core_web_sm

## Utilisation

1️⃣ Préparer et entraîner les modèles
python scripts/download_data.py
python scripts/preprocess.py
python scripts/train_word2vec.py
python scripts/train_classifier.py
python scripts/finetune_transformer.py
python scripts/evaluate.py

2️⃣ Lancer l’application Streamlit
streamlit run streamlit_app.py

## Technologies
- Python (Pandas, NumPy, Scikit-learn)
- NLP : NLTK, spaCy, Gensim
- Deep Learning : PyTorch, Transformers (Hugging Face)
- Visualisation & déploiement : Streamlit

## Résultats attendus

- Comparaison claire entre Word2Vec + Logistic Regression et DistilBERT fine-tuné
- Une interface interactive pour tester le modèle
- Une architecture reproductible et modulaire



