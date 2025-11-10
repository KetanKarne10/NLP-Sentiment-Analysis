import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
from sklearn.metrics import classification_report

@st.cache_resource
def load_models_vectorizer():
    svm = joblib.load("svm_model.pkl")
    logreg = joblib.load("logreg_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return svm, logreg, vectorizer

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

svm, logreg, vectorizer = load_models_vectorizer()

st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Select Model", ("SVM", "Logistic Regression"))
input_mode = st.sidebar.radio("Input Mode", ["Live Input", "Upload Excel"])
show_explainability = st.sidebar.checkbox("Show Top Influential Words (Explainability)", value=True)

st.title("Advanced Sentiment Analysis for Product Reviews")

svm_acc = 0.93
logreg_acc = 0.92

tab_live, tab_batch, tab_model_info = st.tabs(["Live Input", "Batch Upload", "Model Information"])

with tab_live:
    title_text = st.text_input("Review Title")
    body_text = st.text_area("Review Body")
    if st.button("Predict Sentiment (Live)"):
        review_text = title_text + " " + body_text
        if review_text.strip() == "":
            st.warning("Please enter review text to analyze.")
        else:
            X = vectorizer.transform([review_text])
            if model_choice == "SVM":
                pred = svm.predict(X)[0]
                acc = svm_acc
                coefs = svm.coef_[0]
            else:
                pred = logreg.predict(X)[0]
                acc = logreg_acc
                coefs = logreg.coef_[0]

            sentiment = "Positive" if pred == 1 else "Negative"
            if sentiment == "Negative":
                st.markdown(f"<p style='color:red;'>Predicted Sentiment: {sentiment}</p>", unsafe_allow_html=True)
            else:
                st.success(f"Predicted Sentiment: {sentiment}")
            st.markdown(f"**Model Accuracy:** {acc*100:.2f}%")

            # Customer Satisfaction Level
            feature_names = vectorizer.get_feature_names_out()
            coef_dict = dict(zip(feature_names, coefs))
            input_words = set(review_text.lower().split())

            pos_words = [(coef, word) for word, coef in coef_dict.items() if coef > 0 and word in input_words]
            neg_words = [(coef, word) for word, coef in coef_dict.items() if coef < 0 and word in input_words]

            pos_words_sorted = sorted(pos_words, reverse=True)[:10]
            neg_words_sorted = sorted(neg_words)[:10]

            pos_count = len(pos_words_sorted)
            neg_count = len(neg_words_sorted)

            if pos_count > neg_count:
                satisfaction_msg = "Customer sentiment shows **high satisfaction or positive interest** based on your input."
            elif neg_count > pos_count:
                satisfaction_msg = "Customer sentiment indicates **concerns or dissatisfaction** from your input."
            else:
                satisfaction_msg = "Customer sentiment suggests a **mixed or neutral interest** in the product."

            st.markdown("### Customer Satisfaction Level")
            st.info(satisfaction_msg)

            if show_explainability:
                if pos_words_sorted:
                    st.write("Top positive influential words in your input:")
                    st.write([word for coef, word in pos_words_sorted])
                else:
                    st.write("No positive influential words found in your input.")

                if neg_words_sorted:
                    st.write("Top negative influential words in your input:")
                    st.write([word for coef, word in neg_words_sorted])
                else:
                    st.write("No negative influential words found in your input.")

with tab_batch:
    uploaded_file = st.file_uploader("Upload Excel file with 'title', 'body', and 'rating' columns", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'title' not in df.columns or 'body' not in df.columns or 'rating' not in df.columns:
                st.error("Uploaded file must contain 'title', 'body', and 'rating' columns.")
            else:
                df['text'] = df['title'].fillna('') + " " + df['body'].fillna('')
                X = vectorizer.transform(df['text'])
                if model_choice == "SVM":
                    preds = svm.predict(X)
                    acc = svm_acc
                else:
                    preds = logreg.predict(X)
                    acc = logreg_acc
                df['Predicted Sentiment'] = ["Positive" if p == 1 else "Negative" for p in preds]

                def map_rating(r):
                    if r in [1, 2]:
                        return "Negative"
                    elif r in [4, 5]:
                        return "Positive"
                    else:
                        return "Neutral"

                df['Sentiment'] = df['rating'].map(map_rating)

                sentiment_counts = df['Sentiment'].value_counts()
                fig1, ax1 = plt.subplots()
                ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')
                st.pyplot(fig1)

                rating_counts = df['rating'].value_counts().sort_index()
                fig2, ax2 = plt.subplots()
                ax2.bar(rating_counts.index.astype(str), rating_counts.values, color='skyblue')
                ax2.set_xlabel('Rating')
                ax2.set_ylabel('Count')
                ax2.set_title('Distribution of Ratings')
                st.pyplot(fig2)

                satisfaction_counts = df['Predicted Sentiment'].value_counts()
                fig3, ax3 = plt.subplots()
                ax3.pie(satisfaction_counts, labels=satisfaction_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff6666'])
                ax3.axis('equal')
                ax3.set_title('Overall Customer Satisfaction from Predictions')
                st.pyplot(fig3)

                st.dataframe(df[['title', 'body', 'rating', 'Predicted Sentiment']])

                excel_data = to_excel(df[['title', 'body', 'rating', 'Predicted Sentiment']])
                st.download_button(
                    label="Download Results as Excel",
                    data=excel_data,
                    file_name="sentiment_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab_model_info:
    st.subheader("Model Performance Summary")
    acc = svm_acc if model_choice == "SVM" else logreg_acc
    st.markdown(f"### {model_choice} Accuracy: {acc*100:.2f}%")
