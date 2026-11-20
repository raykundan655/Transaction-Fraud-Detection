import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Real-Time Financial Fraud Detection", page_icon="💳")

st.title("Real-Time Financial Fraud Detection")
st.write("Enter transaction details below to check for potential financial fraud.")

MODEL_FILE = "fraud_model.pkl"


@st.cache_resource
def load_pipeline():
    try:
        return joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Error loading model from {MODEL_FILE}: {e}")
        return None


pipeline = load_pipeline()


def engineer_features(df):
    df_feat = df.copy()
    df_feat["orig_balance_diff"] = (
        df_feat["oldbalanceOrg"] - df_feat["newbalanceOrig"] - df_feat["amount"]
    )
    df_feat["dest_balance_diff"] = (
        df_feat["newbalanceDest"] - df_feat["oldbalanceDest"] - df_feat["amount"]
    )
    df_feat["is_orig_zero"] = (df_feat["newbalanceOrig"] == 0).astype(int)
    df_feat["amount_to_oldbalance_orig"] = (
        df_feat["amount"] / (df_feat["oldbalanceOrg"] + 1.0)
    )
    return df_feat


col1, col2 = st.columns(2)

with col1:
    tx_type = st.selectbox(
        "Transaction Type",
        options=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
    )
    tx_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0,
        step=50.0
    )
    tx_step = st.number_input(
        "Transaction Time Step (Hour)",
        min_value=1,
        max_value=744,
        value=1,
        step=1
    )

with col2:
    tx_oldbalanceOrg = st.number_input(
        "Sender Old Balance",
        min_value=0.0,
        value=1000.0,
        step=50.0
    )
    tx_newbalanceOrig = st.number_input(
        "Sender New Balance",
        min_value=0.0,
        value=0.0,
        step=50.0
    )
    tx_oldbalanceDest = st.number_input(
        "Receiver Old Balance",
        min_value=0.0,
        value=0.0,
        step=50.0
    )
    tx_newbalanceDest = st.number_input(
        "Receiver New Balance",
        min_value=0.0,
        value=0.0,
        step=50.0
    )

if st.button("Predict Fraud", type="primary"):
    if pipeline is None:
        st.error("Model is not loaded. Train the model first using fraud_detection_model.py.")
    else:
        try:
            # 1. Create DataFrame from input
            input_data = pd.DataFrame([{
                "step": tx_step,
                "type": tx_type,
                "amount": tx_amount,
                "oldbalanceOrg": tx_oldbalanceOrg,
                "newbalanceOrig": tx_newbalanceOrig,
                "oldbalanceDest": tx_oldbalanceDest,
                "newbalanceDest": tx_newbalanceDest
            }])

            # 2. Engineer features matching the training pipeline
            processed_input = engineer_features(input_data)

            # 3. Predict class and probability
            prediction = pipeline.predict(processed_input)[0]
            probability = pipeline.predict_proba(processed_input)[0, 1]

            st.write("---")

            # 4. Display results
            m1, m2 = st.columns(2)
            with m1:
                result_text = "Fraudulent" if prediction == 1 else "Legitimate"
                st.metric("Prediction", result_text)
            with m2:
                st.metric("Fraud Probability", f"{probability * 100:.2f}%")

            if prediction == 1 or probability >= 0.5:
                st.error("⚠️ This transaction appears suspicious and matches fraudulent patterns.")
            else:
                st.success("✅ This transaction appears legitimate with low risk.")

        except Exception as err:
            st.error(f"An error occurred during prediction: {err}")
