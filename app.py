import streamlit as st
import pandas as pd
import numpy as np
import torch
import re
import html
from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Drug Recommendation System",
    page_icon="💊",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "./801010smua"
DATA_PATH = "./train_with_prediction.csv"

LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Bobot Weighted Scoring
WEIGHT_CONFIDENCE = 0.5
WEIGHT_USEFUL = 0.5


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)
    df["sentiment"] = df["sentiment"].map(LABEL_NAMES)

    return df


tokenizer, model, device = load_model()
df = load_data()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "uniqueID",
    "drugName",
    "condition",
    "review",
    "rating",
    "date",
    "usefulCount",
    "label",
    "sentiment",
    "confidence",
    "prob_negative",
    "prob_neutral",
    "prob_positive"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Kolom berikut tidak ditemukan pada dataset: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# SENTIMENT PREDICTION FUNCTION
# ============================================================

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

def predict_sentiment(text):
    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    predicted_label = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        predicted_label
    ].item()

    return (
        predicted_label,
        confidence,
        probabilities.cpu().numpy()
    )


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def get_recommendations(
    data,
    condition,
    top_n=5
):

    # --------------------------------------------------------
    # 1. Filter berdasarkan condition
    # --------------------------------------------------------

    condition_data = data[
        data["condition"]
        .astype(str)
        .str.lower()
        .str.strip()
        == condition.lower().strip()
    ].copy()

    if len(condition_data) == 0:
        return None, None

    # --------------------------------------------------------
    # 2. Normalisasi usefulCount
    # --------------------------------------------------------

    min_useful = condition_data["usefulCount"].min()
    max_useful = condition_data["usefulCount"].max()

    if max_useful == min_useful:

        condition_data["useful_normalized"] = 1.0

    else:

        condition_data["useful_normalized"] = (
            condition_data["usefulCount"] - min_useful
        ) / (
            max_useful - min_useful
        )

    # --------------------------------------------------------
    # 3. Weighted Recommendation Score
    # --------------------------------------------------------

    def calculate_sentiment_score(row):
        if row["sentiment"] == "Positive":
            return row["confidence"]
        elif row["sentiment"] == "Negative":
            return -row["confidence"]
        else:
            return 0
        
    condition_data["sentiment_score"] = (
        condition_data.apply(
            calculate_sentiment_score,
            axis=1
        )
    )

    condition_data["recommendation_score"] = (
        WEIGHT_CONFIDENCE
        * condition_data["sentiment_score"]
        +
        WEIGHT_USEFUL
        * condition_data["useful_normalized"]
    )

    # --------------------------------------------------------
    # 4. Group berdasarkan drugName
    # --------------------------------------------------------

    drug_scores = (
        condition_data
        .groupby("drugName")
        .agg(
            recommendation_score=(
                "recommendation_score",
                "mean"
            ),
            average_confidence=(
                "confidence",
                "mean"
            ),
            average_usefulCount=(
                "usefulCount",
                "mean"
            ),
            review_count=(
                "review",
                "count"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # 5. Sort recommendation score
    # --------------------------------------------------------

    drug_scores = drug_scores.sort_values(
        by="recommendation_score",
        ascending=False
    )

    # --------------------------------------------------------
    # 6. Top 5
    # --------------------------------------------------------

    top_drugs = drug_scores.head(top_n)

    # --------------------------------------------------------
    # 7. Ambil 1 review setiap sentiment
    # --------------------------------------------------------

    recommendations = []

    for _, drug in top_drugs.iterrows():

        drug_name = drug["drugName"]

        drug_reviews = condition_data[
            condition_data["drugName"] == drug_name
        ]

        st.write("DEBUG - Drug:", drug_name)
        st.write(
            "DEBUG - Sentiment:",
            drug_reviews["sentiment"].value_counts(dropna=False)
        )

        sentiment_reviews = {}

        for sentiment in [
            "Positive",
            "Neutral",
            "Negative"
        ]:

            reviews = drug_reviews[
                drug_reviews["sentiment"]
                .astype(str)
                .str.lower()
                == sentiment.lower()
            ]

            if len(reviews) > 0:

                # Ambil review dengan recommendation
                # score tertinggi pada sentiment tersebut
                selected_review = (
                    reviews
                    .sort_values(
                        by="recommendation_score",
                        ascending=False
                    )
                    .iloc[0]
                )

                sentiment_reviews[sentiment] = (
                    selected_review
                )

            else:

                sentiment_reviews[sentiment] = None

        recommendations.append({
            "drugName": drug_name,
            "recommendation_score":
                drug["recommendation_score"],
            "average_confidence":
                drug["average_confidence"],
            "average_usefulCount":
                drug["average_usefulCount"],
            "review_count":
                drug["review_count"],
            "reviews":
                sentiment_reviews
        })

    return recommendations, condition_data


#================================================================================================================================


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💊 Drug Recommendation")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Analisis Sentimen",
        "Rekomendasi Obat"
    ]
)


# ============================================================
# SENTIMENT ANALYSIS PAGE
# ============================================================

if menu == "Analisis Sentimen":

    st.title("💬 Analisis Sentimen Ulasan Obat")

    st.write(
        "Masukkan ulasan obat untuk mengetahui "
        "kelas sentimen berdasarkan model BERT."
    )

    review_text = st.text_area(
        "Masukkan Ulasan",
        placeholder=(
            "Contoh: This medicine works very well "
            "and I have experienced no side effects."
        ),
        height=180
    )

    predict_button = st.button(
        "🔍 Analisis Sentimen",
        use_container_width=True
    )

    if predict_button:

        if review_text.strip() == "":

            st.warning(
                "Silakan masukkan ulasan terlebih dahulu."
            )

        else:

            with st.spinner(
                "Model BERT sedang melakukan prediksi..."
            ):

                predicted_label, confidence, probabilities = (
                    predict_sentiment(review_text)
                )

            sentiment = LABEL_NAMES[
                predicted_label
            ]

            st.subheader("Hasil Prediksi")

            # ------------------------------------------------
            # Prediction + Confidence
            # ------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Sentiment",
                    sentiment
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            st.divider()

            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            st.subheader(
                "Probabilitas Masing-Masing Kelas"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Negative",
                    f"{probabilities[0] * 100:.2f}%"
                )

                st.progress(
                    float(probabilities[0])
                )

            with col2:

                st.metric(
                    "Neutral",
                    f"{probabilities[1] * 100:.2f}%"
                )

                st.progress(
                    float(probabilities[1])
                )

            with col3:

                st.metric(
                    "Positive",
                    f"{probabilities[2] * 100:.2f}%"
                )

                st.progress(
                    float(probabilities[2])
                )

            st.divider()

            st.caption(
                "Confidence merupakan probabilitas model "
                "terhadap kelas sentimen yang diprediksi."
            )


# ============================================================
# DRUG RECOMMENDATION PAGE
# ============================================================

elif menu == "Rekomendasi Obat":

    st.title("💊 Rekomendasi Obat")

    st.write(
        "Masukkan kondisi penyakit untuk mendapatkan "
        "5 rekomendasi obat berdasarkan ulasan pasien."
    )

    condition_input = st.text_input(
        "Masukkan Kondisi Penyakit",
        placeholder="Contoh: Depression"
    )

    recommend_button = st.button(
        "💊 Cari Rekomendasi",
        use_container_width=True
    )

    if recommend_button:

        if condition_input.strip() == "":

            st.warning(
                "Silakan masukkan kondisi penyakit."
            )

        else:

            with st.spinner(
                "Mencari rekomendasi obat..."
            ):

                recommendations, condition_data = (
                    get_recommendations(
                        df,
                        condition_input,
                        top_n=5
                    )
                )

            if recommendations is None:

                st.error(
                    f"Tidak ditemukan data untuk kondisi "
                    f"'{condition_input}'."
                )

            else:

                st.success(
                    f"Ditemukan {len(condition_data):,} "
                    f"ulasan untuk kondisi "
                    f"'{condition_input}'."
                )

                st.subheader(
                    "🏆 Top 5 Rekomendasi Obat"
                )

                # =================================================
                # DISPLAY TOP 5
                # =================================================

                for rank, recommendation in enumerate(
                    recommendations,
                    start=1
                ):

                    drug_name = recommendation[
                        "drugName"
                    ]

                    score = recommendation[
                        "recommendation_score"
                    ]

                    avg_confidence = recommendation[
                        "average_confidence"
                    ]

                    avg_useful = recommendation[
                        "average_usefulCount"
                    ]

                    review_count = recommendation[
                        "review_count"
                    ]

                    st.markdown(
                        f"## {rank}. {drug_name}"
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "Recommendation Score",
                            f"{score:.4f}"
                        )

                    with col2:

                        st.metric(
                            "Average Confidence",
                            f"{avg_confidence * 100:.2f}%"
                        )

                    with col3:

                        st.metric(
                            "Average Useful Count",
                            f"{avg_useful:.2f}"
                        )

                    with col4:

                        st.metric(
                            "Jumlah Review",
                            f"{review_count:,}"
                        )

                    # =================================================
                    # REVIEWS
                    # =================================================

                    st.markdown(
                        "**Contoh Ulasan Berdasarkan Sentimen**"
                    )

                    reviews = recommendation[
                        "reviews"
                    ]

                    # -------------------------------------------------
                    # Positive
                    # -------------------------------------------------

                    positive_review = reviews[
                        "Positive"
                    ]

                    if positive_review is not None:

                        with st.expander(
                            "🟢 Ulasan Positive"
                        ):

                            st.write(
                                positive_review["review"]
                            )

                            st.caption(
                                f"Rating: "
                                f"{positive_review['rating']} | "
                                f"Useful: "
                                f"{positive_review['usefulCount']} | "
                                f"Confidence: "
                                f"{positive_review['confidence']:.4f}"
                            )

                    else:

                        st.info(
                            "Tidak tersedia ulasan Positive."
                        )

                    # -------------------------------------------------
                    # Neutral
                    # -------------------------------------------------

                    neutral_review = reviews[
                        "Neutral"
                    ]

                    if neutral_review is not None:

                        with st.expander(
                            "🟡 Ulasan Neutral"
                        ):

                            st.write(
                                neutral_review["review"]
                            )

                            st.caption(
                                f"Rating: "
                                f"{neutral_review['rating']} | "
                                f"Useful: "
                                f"{neutral_review['usefulCount']} | "
                                f"Confidence: "
                                f"{neutral_review['confidence']:.4f}"
                            )

                    else:

                        st.info(
                            "Tidak tersedia ulasan Neutral."
                        )

                    # -------------------------------------------------
                    # Negative
                    # -------------------------------------------------

                    negative_review = reviews[
                        "Negative"
                    ]

                    if negative_review is not None:

                        with st.expander(
                            "🔴 Ulasan Negative"
                        ):

                            st.write(
                                negative_review["review"]
                            )

                            st.caption(
                                f"Rating: "
                                f"{negative_review['rating']} | "
                                f"Useful: "
                                f"{negative_review['usefulCount']} | "
                                f"Confidence: "
                                f"{negative_review['confidence']:.4f}"
                            )

                    else:

                        st.info(
                            "Tidak tersedia ulasan Negative."
                        )

                    st.divider()