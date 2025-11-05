# scripts/finetune_transformer.py
"""
Fine-tune DistilBERT (distilbert-base-uncased) sur IMDB
Utilise Hugging Face Trainer API. Sauvegarde modèle dans models/distilbert_finetuned
"""


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os

def main():
    # Détection de l'appareil
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Chargement du dataset IMDB
    dataset = load_dataset("imdb")
    
    # Pour accélérer le test, on prend un subset
    train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(500))

    # Modèle et tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Tokenization
    def preprocess(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)
    
    train_dataset = train_dataset.map(preprocess, batched=True)
    test_dataset = test_dataset.map(preprocess, batched=True)

    # Set format pour PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir="./models/distilbert_finetuned",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Fine-tuning
    trainer.train()

    # Évaluation
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # Sauvegarde du modèle et tokenizer
    model_dir = "./models/distilbert_finetuned_final"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Modèle et tokenizer sauvegardés dans {model_dir}")

if __name__ == "__main__":
    main()
