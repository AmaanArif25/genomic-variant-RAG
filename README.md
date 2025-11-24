# Genomic Variant Classification & RAG Pipeline

This repository contains two integrated assignments:

- **Assignment 1**: Fine-tuning BioBERT for genomic variant classification
- **Assignment 2**: Building a Genomic-RAG pipeline using Pinecone, embeddings, and Streamlit

Both systems work independently but can also be combined for a hybrid workflow.

---

## 🚀 Project Overview

This project provides:

### ✅ Assignment 1 — BioBERT Variant Classifier
- Fine-tuning BioBERT on variant–disease association text
- Training, evaluation, saving models
- Command-line prediction interface

### ✅ Assignment 2 — Genomic RAG System
- Embedding genomic variants
- Storing & retrieving them using Pinecone
- Generating responses based on retrieved evidence
- Fully interactive Streamlit app

---

## 📂 Repository Structure
```
project/
│
├── Assignment1/
│   ├── train_biobert.py
│   ├── predict.py
│   ├── requirements.txt
│   └── variants.json
│
├── Assignment2/
│   ├── app.py
│   ├── main.py
│   ├── config.py
│   ├── data/variants.json
│   ├── src/
│   │   ├── embedding_manager.py
│   │   ├── retrieval.py
│   │   ├── generation.py
│   │   └── evaluation.py
│   └── requirements.txt
│
└── README.md  (this file)
```

---

## 🧪 ASSIGNMENT 1 — BioBERT Variant Classification

### 1️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2️⃣ Prepare Training Data

Place the JSON dataset as:
```
Assignment1/variants.json
```

### 3️⃣ Train the Model

**Basic Training**
```bash
python train_biobert.py
```

This will:
- Load & preprocess dataset
- Split into train/val/test
- Fine-tune BioBERT for 3 epochs
- Evaluate results
- Save the trained model

**Example Training Log**
```
📂 Loading dataset...
✅ Loaded 70 samples
📊 Unique labels: 10
🚀 Starting model training...
Epoch 1/3: Training...
Epoch 2/3: Training...
Epoch 3/3: Training...
🎯 Test Accuracy: 0.906 (90.6%)
```

**Expected Training Time**

| Hardware | Duration |
|----------|----------|
| CPU      | 30–45 minutes |
| GPU      | 5–10 minutes |

### 4️⃣ Make Predictions

**Interactive Mode**
```bash
python predict.py --interactive
```

Example:
```
Enter query: EGFR L858R in lung
Enter query: TP53 p.R248W in breast
```

**Single Query Mode**
```bash
python predict.py --query "BRCA1 c.5266dupC in ovarian?"
```

---

## 🧬 ASSIGNMENT 2 — Genomic RAG Pipeline (Pinecone + Embeddings + Streamlit)

### 📦 Part 2: Install Dependencies (3 minutes)

**2.1 Create & Install Requirements**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**2.2 Verify Installation**
```bash
python -c "import torch; import pinecone; import streamlit; print('✓ All packages installed')"
```

### 🌲 Part 3: Pinecone Setup (3 minutes)

**3.1 Create Pinecone Account**
1. Go to https://www.pinecone.io
2. Sign up → verify email

**3.2 Generate API Key**
1. Open API Keys
2. Click Create Key
3. Copy the key

**3.3 Add to .env**

Create:
```env
PINECONE_API_KEY=your-real-key
PINECONE_ENVIRONMENT=gcp-starter
```

### 🧠 Part 4: Project Files

Create the following files:
- `config.py`
- `src/embedding_manager.py`
- `src/retrieval.py`
- `src/generation.py`
- `src/evaluation.py`
- `main.py`
- `app.py`
- `data/variants.json`

### ⚙️ Part 5: Initialize the RAG Pipeline

**5.1 Setup Database**
```bash
python main.py --setup
```

**Expected Output**
```
================================================================================
GENOMIC RAG PIPELINE - SETUP
================================================================================

[1/3] Initializing Embedding Manager...
Loading embedding model...
Initializing Pinecone...

[2/3] Creating Pinecone Index...
Creating index: variants-index
Index ready!

[3/3] Loading and Indexing Variants...
Loading variants from data/variants.json...
Loaded 20 variants
Processing 20 variants for upsert...
100%|██████████████████████████████████| 20/20
✓ Successfully upserted 20 variants to Pinecone

================================================================================
✓ SETUP COMPLETE
================================================================================
Total variants indexed: 20
Index name: variants-index
Embedding dimension: 384
```

**5.2 Run a Query**
```bash
python main.py --query "Best treatment for BRCA1 mutations?"
```

You will see:
- Top retrieved variants
- Generated answer

This launches a web UI where you can:
- Submit genomic queries
- View retrieved variants
- View generated insights

---

## 📧 Contact

For questions or support, please open an issue in the repository.
