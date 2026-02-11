import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("🎬 IMDB Sentiment Analysis")

# ---------------- Upload CSV ----------------
uploaded_file = st.file_uploader("Upload IMDB Dataset CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Convert sentiment to numeric
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # ---------------- Train Test Split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'],
        test_size=0.2,
        random_state=42
    )

    # ---------------- TF-IDF ----------------
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # ---------------- Model ----------------
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)

    y_pred = nb.predict(X_test_tfidf)

    # ---------------- Accuracy ----------------
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: **{accuracy:.4f}**")

    # ---------------- Confusion Matrix ----------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted Negative", "Predicted Positive"],
        index=["Actual Negative", "Actual Positive"]
    )

    st.dataframe(cm_df)

    # ---------------- Sentiment Distribution Graph ----------------
    st.subheader("Sentiment Distribution")

    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    # ---------------- Test Review ----------------
    st.subheader("Test a Review")

    user_review = st.text_input("Enter a Review")

    if user_review:
        vec = tfidf.transform([user_review])
        prediction = nb.predict(vec)[0]

        if prediction == 1:
            st.success("Sentiment: Positive 😊")
        else:
            st.error("Sentiment: Negative 😞")
