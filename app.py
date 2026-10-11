!pip install transformers datasets optuna scikit-learn

import pandas as pd
import numpy as np
import re
import html
import torch
import time
import optuna
import torch.nn.functional as F
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import softmax
from wordcloud import WordCloud
from collections import Counter
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==============================================================================
# 1. Perolehan Dataset
# ==============================================================================
train_df = pd.read_csv("/kaggle/input/datasets/jessicali9530/kuc-hackathon-winter-2018/drugsComTrain_raw.csv")
test_df = pd.read_csv("/kaggle/input/datasets/jessicali9530/kuc-hackathon-winter-2018/drugsComTest_raw.csv")
df = pd.concat([train_df, test_df], ignore_index=True)       # Gabungkan dataset 

# ==============================================================================
# 2. EDA - a) Pengecekan Jumlah Baris dan Kolom
# ==============================================================================
rows, columns = df.shape
print("Jumlah baris   :", rows)
print("Jumlah kolom   :", columns)

# ==============================================================================
# 2. EDA - b) Pengecekan Tipe Data
# ==============================================================================
print(df.dtypes)

# ==============================================================================
# 2. EDA - c) Pengecekan Jumlah Unique drugName dan condition
# ==============================================================================
print("Jumlah unique drugName :", df["drugName"].nunique())
print("Jumlah unique condition:", df["condition"].nunique())
print("DrugName terbanyak:")
print(df["drugName"].value_counts().head(10))
print("\nCondition terbanyak:")
print(df["condition"].value_counts().head(10))

# ==============================================================================
# 2. EDA - d) Pengecekan Data Null dan Data Duplikat
# ==============================================================================
print(df.isnull().sum())                        # Jumlah data null setiap kolom
print("\nTotal data duplikat:", df.duplicated().sum())   # Jumlah baris duplikat

# ==============================================================================
# 2. EDA - e) Pengecekan Panjang Kata Ulasan
# ==============================================================================
# 1) Menghitung panjang review berdasarkan jumlah kata
Word_count = df["review"].fillna("").str.split().str.len()
print("Statistik Panjang Review:")              # Statistik deskriptif
print(word_count.describe())
total_data = len(df)                            # Jumlah data
count_over_128 = (word_count > 128).sum()       # Data dengan panjang > 128 kata
percent_over_128 = (count_over_128 / total_data) * 100
count_over_256 = (word_count > 256).sum()       # Data dengan panjang > 256 kata
percent_over_256 = (count_over_256 / total_data) * 100
# Tampilkan hasil
print(f"Review >128 kata   : {count_over_128} ({percent_over_128:.2f}%)")
print(f"Review >256 kata   : {count_over_256} ({percent_over_256:.2f}%)")

# ==============================================================================
# 3. Preprocessing, Labeling, Tokenization, Splitting - a) PREPROCESSING
# ==============================================================================
# text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
# text = re.sub(r'[^a-zA-Z0-9\s\.,]', ' ', text)
def clean_text(text):
    text = html.unescape(str(text))
    text = text.replace('"', '').replace("'", '')
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.,]', ' ', text)
    text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

df["review"] = df["review"].apply(clean_text)

# ==============================================================================
# 3. Preprocessing, Labeling, Tokenization, Splitting - b) LABELING
# ==============================================================================
def rating_to_label(rating):
    if 1 <= rating <= 4:
        return 0
    elif 5 <= rating <= 7:
        return 1
    else:
        return 2

df["label"] = df["rating"].apply(rating_to_label)

# ==============================================================================
# 2. EDA - f) Analisis Frekuensi Kemunculan Kata
# ==============================================================================
# 1) Fungsi untuk mendapatkan frekuensi kata
def get_word_frequency(texts):
    text = " ".join(texts.astype(str))                 # Gabungkan seluruh teks
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text) #Ambil hanya kata/alphanumeric
    word_freq = Counter(words)                         # Hitung frekuensi
    return word_freq

# 2) Fungsi membuat WordCloud
def create_wordcloud(word_freq, title):
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=100,
        collocations=False
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=20)
    plt.show()

# 3) FREKUENSI KATA DATA KESELURUHAN
freq_all = get_word_frequency(df["review"])
print("=== 20 Kata Paling Sering Muncul - Keseluruhan ===")
for word, count in freq_all.most_common(20):
    print(f"{word:20s} : {count:,}")
#---WordCloud keseluruhan
create_wordcloud(
    freq_all,
    "WordCloud Kata pada Seluruh Data"
)

# 4) FREKUENSI KATA SENTIMEN POSITIF
positive_df = df[df["label"] == 2]
freq_positive = get_word_frequency(positive_df["review"])
print("\n=== 20 Kata Paling Sering Muncul - Sentimen Positif ===")
for word, count in freq_positive.most_common(20):
    print(f"{word:20s} : {count:,}")
#---WordCloud positif
create_wordcloud(
    freq_positive,
    "WordCloud Sentimen Positif"
)

# 5) FREKUENSI KATA SENTIMEN NETRAL
neutral_df = df[df["label"] == 1]
freq_neutral = get_word_frequency(neutral_df["review"])
print("\n=== 20 Kata Paling Sering Muncul - Sentimen Netral ===")
for word, count in freq_neutral.most_common(20):
    print(f"{word:20s} : {count:,}")
#---WordCloud netral
create_wordcloud(
    freq_neutral,
    "WordCloud Sentimen Netral"
)

# 6) FREKUENSI KATA SENTIMEN NEGATIF
negative_df = df[df["label"] == 0]
freq_negative = get_word_frequency(negative_df["review"])
print("\n=== 20 Kata Paling Sering Muncul - Sentimen Negatif ===")
for word, count in freq_negative.most_common(20):
    print(f"{word:20s} : {count:,}")
#---WordCloud negatif
create_wordcloud(
    freq_negative,
    "WordCloud Sentimen Negatif"
)
# ==============================================================================
# 3. Preprocessing, Labeling, Tokenization, Splitting - c) SPLITTING DATA
# ==============================================================================
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=df["label"] 
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    shuffle=True,
    stratify=temp_df["label"] 
)

# ==============================================================================
# 3. Preprocessing, Labeling, Tokenization, Splitting - d) TOKENIZATION
# ==============================================================================
train_texts = train_df["review_clean"]
train_labels = train_df["label"]
val_texts = val_df["review_clean"]
val_labels = val_df["label"]
test_texts = test_df["review_clean"]
test_labels = test_df["label"]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=256
    )

train_enc = tokenize(train_texts)
val_enc = tokenize(val_texts)
test_enc = tokenize(test_texts)

train_dataset = Dataset.from_dict({
    "input_ids": train_enc["input_ids"],
    "attention_mask": train_enc["attention_mask"],
    "labels": train_labels.tolist()
})

val_dataset = Dataset.from_dict({
    "input_ids": val_enc["input_ids"],
    "attention_mask": val_enc["attention_mask"],
    "labels": val_labels.tolist()
})

test_dataset = Dataset.from_dict({
    "input_ids": test_enc["input_ids"],
    "attention_mask": test_enc["attention_mask"],
    "labels": test_labels.tolist()
})

def compute_metrics(eval_pred):                           
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

# ==============================================================================
# 2. EDA - e) Pengecekan Panjang Kata Ulasan
# ==============================================================================
# 2) Menghitung panjang review berdasarkan jumlah token
Word_count = df["review"].fillna("").apply(lambda x: len(tokenizer.tokenize(str(x))))
print("Statistik Panjang Review:")              # Statistik deskriptif
print(df["token_length"].describe())
count_over_128 = (word_count > 128).sum()       # Data dengan panjang > 128 kata
percent_over_128 = (count_over_128 / total_data) * 100
count_over_256 = (word_count > 256).sum()       # Data dengan panjang > 256 kata
percent_over_256 = (count_over_256 / total_data) * 100
# Tampilkan hasil
print(f"Review >128 kata   : {count_over_128} ({percent_over_128:.2f}%)")
print(f"Review >256 kata   : {count_over_256} ({percent_over_256:.2f}%)")

# ==============================================================================
# 4. Bayesian Optimization
# ==============================================================================
subset_size = 5000
train_subset = train_texts.iloc[:subset_size]
val_subset = val_texts.iloc[:1000]
train_subset_labels = train_labels.iloc[:subset_size]
val_subset_labels = val_labels.iloc[:1000]

def objective(trial):
    start_time = time.time()
    print(f"\nStarting trial {trial.number}")

    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    )

    train_enc = tokenize(train_subset)
    val_enc = tokenize(val_subset)
    train_dataset = Dataset.from_dict({
        "input_ids": train_enc["input_ids"],
        "attention_mask": train_enc["attention_mask"],
        "label": train_subset_labels.tolist()
    })
    val_dataset = Dataset.from_dict({
        "input_ids": val_enc["input_ids"],
        "attention_mask": val_enc["attention_mask"],
        "label": val_subset_labels.tolist()
    })

    training_args = TrainingArguments(
        output_dir="./optuna_results",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=weight_decay,
        logging_steps=100,
        disable_tqdm=False,
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_result = trainer.evaluate()
    end_time = time.time()
    print(f"Trial {trial.number} finished in {(end_time-start_time)/60:.2f} minutes")
    return eval_result["eval_accuracy"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# ==============================================================================
# 5. Training Model BERT dengan Strategi Oversampling Berdasarkan Hard Samples
# ==============================================================================
best_params = study.best_params
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

training_args = TrainingArguments(
    output_dir="./final_model",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=best_params["lr"],
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    weight_decay=best_params["weight_decay"],
    num_train_epochs=1,                   # hanya 1 epoch
    logging_strategy="epoch",
    logging_steps=None,
    report_to="none",
    fp16=True,
    eval_accumulation_steps=10,           # KUNCI: pindah logits ke CPU bertahap
    dataloader_pin_memory=False,          # kurangi overhead memori host->GPU
)

original_train_df = train_df.copy()
original_enc = tokenize(original_train_df["review"])
original_dataset = Dataset.from_dict({
    "input_ids": original_enc["input_ids"],
    "attention_mask": original_enc["attention_mask"],
    "labels": original_train_df["label"].tolist()
})

del original_enc                             
gc.collect()                                 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

def predict_train_metrics(trainer, dataset):
    pred = trainer.predict(dataset)
    logits = pred.predictions
    labels = pred.label_ids
    probs = softmax(logits, axis=1)
    preds = np.argmax(logits, axis=1)
    train_acc = accuracy_score(labels, preds)
    train_loss = log_loss(labels, probs)            #loss konsisten dg train_acc
    confidence = probs[np.arange(len(labels)), labels]
    # bersihkan hasil prediksi besar
    del pred, logits, probs, preds                       
    gc.collect()                                         
    torch.cuda.empty_cache()                             
    return train_acc, train_loss, confidence

def make_new_dataset(original_df, confidence, threshold=0.6):
    hard_idx = np.where(confidence < threshold)[0]
    print(f"Hard samples : {len(hard_idx)}")
    hard_df = original_df.iloc[hard_idx].copy()
    new_df = pd.concat(
        [original_df, hard_df],
        ignore_index=True
    )
    return new_df

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

print("\nStarting Final Training...")
start_time = time.time()
num_epochs = 5
current_df = original_train_df.copy()

for epoch in range(num_epochs):
    print("="*60)
    print(f"Epoch {epoch+1}")
    print(f"Training size : {len(current_df)}")

    ##################################################
    # Buat dataset training
    ##################################################
    enc = tokenize(current_df["review"])
    train_dataset = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": current_df["label"].tolist()
    })
    del enc
    gc.collect()
    trainer.train_dataset = train_dataset
    ##################################################
    #Reset optimizer &scheduler tiap epoch spy state Adam lama tdk numpuk di GPU
    ##################################################
    trainer.optimizer = None
    trainer.lr_scheduler = None
    gc.collect()
    torch.cuda.empty_cache()

    ##################################################
    # Training 1 epoch
    ##################################################
    trainer.train(resume_from_checkpoint=False)
    gc.collect()
    torch.cuda.empty_cache()
    print("Training selesai.")

    ##################################################
    # Evaluasi VALIDATION
    ##################################################
    print("Evaluasi validation...")
    val_eval = trainer.evaluate(val_dataset)
    val_loss_history.append(val_eval["eval_loss"])
    val_acc_history.append(val_eval["eval_accuracy"])
    del val_eval
    gc.collect()
    torch.cuda.empty_cache()

    ###################################################
    # Train Accuracy + Confidence
    ###################################################
    print("Menghitung confidence pada train...")
    train_acc, train_loss_full, confidence = predict_train_metrics(trainer, original_dataset)
    train_loss_history.append(train_loss_full) #ganti smbrnya,bkn dr log_history
    train_acc_history.append(train_acc)
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Mean Confidence : {confidence.mean():.4f}")

    ###################################################
    # Dataset baru
    ###################################################
    print("Membuat dataset baru...")
    current_df = make_new_dataset(original_train_df, confidence, threshold=0.6)
    print(f"Next Training Size : {len(current_df)}")
    del train_dataset, confidence
    gc.collect()
    torch.cuda.empty_cache()

end_time = time.time()
print(f"Final Training Time: {(end_time-start_time)/60:.2f} minutes")

# ==============================================================================
# 6. Evaluasi Model
# ==============================================================================
preds_output_test = trainer.predict(test_dataset)
logits_test = preds_output_test.predictions
true_labels_test = preds_output_test.label_ids
preds_test = np.argmax(logits_test, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(true_labels_test, preds_test, digits=4))

# Hitung confusion matrix
cm = confusion_matrix(true_labels_test, preds_test)
class_names = ["Negative", "Neutral", "Positive"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Test Set")
plt.savefig("confusion_matrix_test.png", dpi=150, bbox_inches="tight")
plt.show()

# ==============================================================================
# 7. Sistem Rekomendasi Obat Dengan Content-Based Filtering
# ==============================================================================
# 1) Siapkan Dataset
df_texts = df["review"]
df_labels = df["label"]
df_enc = tokenize(df_texts)
df_dataset = Dataset.from_dict({
    "input_ids": df_enc["input_ids"],
    "attention_mask": df_enc["attention_mask"],
    "labels": df_labels.tolist()
})
dataset_pred = df_dataset

2) Prediksi Dataset
preds_output = trainer.predict(dataset_pred)
logits = preds_output.predictions                   # Logits
probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()

3) Simpan Hasil Prediksi
prob_negative = probabilities[:, 0]
prob_neutral = probabilities[:, 1]
prob_positive = probabilities[:, 2]
predicted_labels = np.argmax(probabilities, axis=1)

result_df = df.copy()
result_df["sentiment"] = predicted_labels
result_df["prob_negative"] = prob_negative
result_df["prob_neutral"] = prob_neutral
result_df["prob_positive"] = prob_positive
result_df["confidence"] = probabilities.max(axis=1)       #conf= probs tertinggi

4) Membuat Sentiment Score
def calculate_sentiment_score(row):
    if row["sentiment"] == 2:         # Positive
        return row["confidence"]
    elif row["sentiment"] == 0:       # Negative
        return -row["confidence"]
    else:                             # Neutral
        return 0
df["sentiment_score"] = df.apply(calculate_sentiment_score, axis=1)

5) Normalisasi usefulCount
scaler = MinMaxScaler()
df["useful_score"] = scaler.fit_transform(df[["usefulCount"]])

6) Menggabungkan Nilai Sentiment dan usefulCount
alpha = 0.8
beta = 0.2
df["recommendation_score"] = (
    alpha * df["sentiment_score"] + beta * df["useful_score"]
)

7) Content-based filtering
def recommend_drugs(
    condition,
    data=df,
    top_n=5
):

    # Filter berdasarkan condition
    candidates = data[
        data["condition"].str.lower() == condition.lower()
    ].copy()

    if len(candidates) == 0:
        return None

    # Hitung skor rata-rata setiap obat
    recommendations = (
        candidates
        .groupby("drugName")
        .agg(
            # sbg acuan utk peringkat
            recommendation_score=("recommendation_score", "mean"),
            # menghitung rata2 sentimen dr suatu obat
            sentiment_score=("sentiment_score", ”mean"),
            # tau jmlh user yg menyetujui review obat
            usefulCount=(“usefulCount", "sum"),
            # tau sbrp byk obat diulas
            review_count=("review", "count")
        )
        .reset_index()
    )

    # Ranking
    recommendations = recommendations.sort_values(
        by="recommendation_score",
        ascending=False
    )

    return recommendations.head(top_n)

# Penggunaan
recommendations = recommend_drugs(
    condition="Depression",
    top_n=5
)
print(recommendations)
