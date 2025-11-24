import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import pickle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

class VariantDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def augment_data(df):
    """Augment training data with paraphrased queries"""
    print("Augmenting data...")
    
    augmented_rows = []
    
    # Augmentation templates
    templates = [
        "Query: {variant} in {cancer_type}?",
        "What about {variant} in {cancer_type}?",
        "Variant {variant} found in {cancer_type}?",
        "{variant} detected in {cancer_type}?",
        "Query: {variant} variant in {cancer_type}?",
    ]
    
    for _, row in df.iterrows():
        # Original
        augmented_rows.append(row)
        
        # Add augmented versions
        for template in templates[1:]:  # Skip first (original format)
            aug_row = row.copy()
            aug_row['query'] = template.format(
                variant=row['variant'],
                cancer_type=row['cancer_type']
            )
            augmented_rows.append(aug_row)
    
    aug_df = pd.DataFrame(augmented_rows)
    print(f"Augmented from {len(df)} to {len(aug_df)} samples\n")
    
    return aug_df

def load_and_prepare_data(json_file_path):
    """Load and prepare the variant dataset"""
    print("Loading dataset...")
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Create input queries
    df['query'] = df.apply(
        lambda row: f"Query: {row['variant']} in {row['cancer_type']}?",
        axis=1
    )
    
    df['label_text'] = df['interpretation']
    
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label_text'])
    
    print(f"Loaded {len(df)} samples")
    print(f"Unique labels: {len(label_encoder.classes_)}\n")
    
    return df, label_encoder

def create_splits_with_augmentation(df):
    """Split data and augment training set"""
    print("Splitting dataset...")
    
    # Split BEFORE augmentation to avoid data leakage
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"Train (before augmentation): {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples\n")
    
    # Augment ONLY training data
    train_df = augment_data(train_df)
    
    print(f"Train (after augmentation): {len(train_df)} samples\n")
    
    return train_df, val_df, test_df

def tokenize_data(train_df, val_df, test_df, model_name):
    """Tokenize datasets"""
    print("Tokenizing data...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_encodings = tokenizer(train_df['query'].tolist(), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    val_encodings = tokenizer(val_df['query'].tolist(), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    test_encodings = tokenizer(test_df['query'].tolist(), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    
    train_dataset = VariantDataset(train_encodings, train_df['label'].values)
    val_dataset = VariantDataset(val_encodings, val_df['label'].values)
    test_dataset = VariantDataset(test_encodings, test_df['label'].values)
    
    print("Tokenization complete\n")
    
    return tokenizer, train_dataset, val_dataset, test_dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {'accuracy': accuracy_score(labels, preds)}

def train_model(train_dataset, val_dataset, num_labels, model_name, output_dir='./biobert_variant_model'):
    """Train with optimized settings for augmented data"""
    print("Starting training with augmented data...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        save_total_limit=2,
        fp16=False,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Training in progress...\n")
    trainer.train()
    print("\nTraining complete!\n")
    
    return trainer, model

def evaluate_model(trainer, test_dataset, test_df, label_encoder):
    """Evaluate model"""
    print("Evaluating on test set...")
    
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    accuracy = accuracy_score(labels, preds)
    
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    print(classification_report(labels, preds, target_names=label_encoder.classes_, zero_division=0))
    
    return accuracy

def main():
    print("\n" + "=" * 60)
    print("BioBERT with Data Augmentation")
    print("=" * 60 + "\n")
    
    JSON_FILE = 'variants.json'
    MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
    OUTPUT_DIR = './biobert_variant_model'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    try:
        df, label_encoder = load_and_prepare_data(JSON_FILE)
        train_df, val_df, test_df = create_splits_with_augmentation(df)
        tokenizer, train_dataset, val_dataset, test_dataset = tokenize_data(train_df, val_df, test_df, MODEL_NAME)
        trainer, model = train_model(train_dataset, val_dataset, len(label_encoder.classes_), MODEL_NAME, OUTPUT_DIR)
        accuracy = evaluate_model(trainer, test_dataset, test_df, label_encoder)
        
        print("\nSaving model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        with open(f'{OUTPUT_DIR}/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"Model saved to {OUTPUT_DIR}\n")
        
        print("=" * 60)
        if accuracy >= 0.75:
            print(f"SUCCESS! Achieved {accuracy*100:.2f}% accuracy")
        else:
            print(f"Achieved {accuracy*100:.2f}% accuracy")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"Error: {str(e)}\n")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()