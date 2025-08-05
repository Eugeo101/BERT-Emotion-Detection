# Emotion Detection using Transfer Learning on BERT

This project focuses on building an accurate **Emotion Detection system** using **transfer learning** on a pretrained **BERT model**. The dataset used is sourced from the [Hugging Face Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion), which contains labeled text samples across multiple emotion categories.

---

## 🧠 Problem Statement

Detect the **emotion expressed in text**, such as joy, anger, fear, sadness, love, or surprise. The task is treated as a **multi-class classification** problem.

---

## 🚀 Approach

### ✅ Phase 1: Transfer Learning with BERT

- Loaded the **pretrained BERT model** (`bert-base-uncased`) and its tokenizer from Hugging Face.
- Extracted the **hidden state of the `[CLS]` token** as fixed-length feature vectors.
- Trained a traditional **machine learning model** (e.g., Logistic Regression / Random Forest / SVM) on the embeddings.
- Initial **error analysis** revealed misclassifications between closely related emotions (e.g., *surprise* vs *fear*).

### 🔁 Phase 2: Fine-tuning BERT

- Used the `Trainer` API from `transformers` to **fine-tune BERT end-to-end** with a classification head.
- Achieved strong performance on the test set:
  - **Accuracy**: `92.6%`
  - **F1 Score**: `92.6%`

---

## 📊 Error Analysis

- Performed **confusion matrix analysis** to identify frequent errors.
- Found **mislabeled data** in the original dataset (e.g., texts labeled *joy* but expressing *fear*).
- Common confusion pairs included:
  - **Surprise vs Fear**: Some expressions of surprise had fearful tones.
  - **Fear vs Anger**: Emotional overlap led to incorrect predictions.
- These findings suggest room for improvement through better data labeling and class balancing.

---

## 📌 Future Work

- **Label Correction**: Manually review and correct ambiguous or incorrect emotion labels.
- **Data Expansion**:
  - Collect more examples, especially for *surprise* and *fear*.
  - Include more nuanced emotional expressions to improve model sensitivity.
- **Model Improvements**:
  - Experiment with larger or multilingual BERT variants.
  - Incorporate multi-modal inputs (e.g., voice tone or facial expression).

---

## 🛠️ Tools & Libraries

- [🤗 Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- PyTorch
- Scikit-learn
- Matplotlib / Seaborn (for visualization)

---

## 📁 Project Structure

