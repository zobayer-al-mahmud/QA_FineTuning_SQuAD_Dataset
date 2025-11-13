# 📘 Fine-Tuning BERT on SQuAD v1.1

### Question Answering with Hugging Face Transformers — A Complete Colab Project

This repository contains a full end-to-end implementation of **fine-tuning a BERT model** for **extractive Question Answering (QA)** using the **SQuAD v1.1** dataset.
The project includes dataset exploration, tokenization, training, evaluation (EM/F1), and multiple real-world test queries.

---

## 🚀 Project Overview

This project demonstrates how to:

* Fine-tune **BERT-Base-Uncased** for Extractive QA
* Use **Hugging Face Transformers + Datasets**
* Train on **Google Colab** with GPU
* Compute **Exact Match (EM)** and **F1 Score** using `evaluate`
* Perform QA inference using a custom `get_answer()` function
* Run multiple custom QA test cases

The project follows an assignment structure and maintains a clear, academic-style workflow.

---

## 📂 Contents

```
📦 SQuAD-BERT-QA
├── notebook.ipynb            # Full Colab notebook
├── qa_bert_finetuned_final/  # Saved final model (weights, tokenizer)
├── README.md                 # Project documentation
└── example_results/          # Test outputs (optional)
```

---

## 📚 Dataset

**SQuAD v1.1** — Stanford Question Answering Dataset
Loaded directly through Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("squad")
```

* **Train samples:** 88,524
* **Validation samples:** 10,784

Each example contains:

```json
{
  "id": "...",
  "title": "...",
  "context": "...",
  "question": "...",
  "answers": {
    "text": [...],
    "answer_start": [...]
  }
}
```

---

## 🧠 Model Used

**BERT-Base-Uncased (110M parameters)**
Loaded with:

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

---

## ⚙️ Fine-Tuning Setup

### **TrainingArguments**

```python
training_args = TrainingArguments(
    output_dir="qa_bert_finetuned",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=100
)
```

### **Trainer API**

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
)
trainer.train()
```

Due to Colab storage limits, final model saving is done **manually** after training by removing weight-heavy checkpoints.

---

## 📈 Evaluation

### 🔹 Metric Used

**SQuAD EM/F1 metric via `evaluate`**

### 🔹 Sample Metric Demonstration

```python
metric = evaluate.load("squad")
results = metric.compute(predictions=sample_predictions, references=sample_references)
```

### 🔹 Full Validation Evaluation

Using feature-to-example mapping, offset mapping, and span extraction.

Expected performance for BERT-Base on SQuAD v1.1:

| Metric               | Typical Range |
| -------------------- | ------------- |
| **Exact Match (EM)** | 90% – 100%     |
| **F1 Score**         | 90% – 100%     |

---

## 🧪 Custom QA Tests

A set of custom evaluation examples were tested using:

```python
answer = get_answer(question, context)
```

### **Examples Tested**

#### 📌 Test 1 — Assignment Example

**Q:** Who developed the theory of relativity?
**A:** Albert Einstein

---

#### 📌 Test 2 — Real SQuAD validation sample

Works across long contexts and overlapping spans.

---

#### 📌 Test 3 — General Knowledge

**Q:** What is the capital of France?
**A:** Paris

---

#### 📌 Test 4 — History

**Q:** Who was the first President of the United States?
**A:** George Washington

---

#### 📌 Test 5 — SQuAD Style

**Q:** What is the main ingredient in guacamole?
**A:** avocados

---

#### 📌 Test 6 — Long Context Stress Test

Handles multi-sentence reasoning in long passages.

---

## 💾 Saved Model

After training:

```python
trainer.save_model("./qa_bert_finetuned_final")
tokenizer.save_pretrained("./qa_bert_finetuned_final")
```

Folder contains:

```
config.json
model.safetensors
tokenizer_config.json
special_tokens_map.json
vocab.txt
```

This model can be loaded via:

```python
model = AutoModelForQuestionAnswering.from_pretrained("./qa_bert_finetuned_final")
tokenizer = AutoTokenizer.from_pretrained("./qa_bert_finetuned_final")
```

---

## 🖥️ How to Run on Colab

1. Clone or upload the notebook
2. Enable GPU
3. Install dependencies:

```bash
pip install transformers datasets evaluate
```

4. Run the notebook
5. Save the final model

---

## 🎓 Learning Outcomes

Through this project, I learned:

* Differences between **classification** and **extractive QA**
* How to preprocess long context passages for QA
* How Hugging Face tokenizers map answer spans to token positions
* How to fine-tune a transformer model on a large dataset
* How to compute **EM/F1** using `evaluate`
* How to perform inference with a trained QA model

---

## 🤝 Acknowledgements

* Hugging Face Transformers
* Hugging Face Datasets
* Stanford SQuAD Team
* Google Colab GPU compute



Just tell me!
