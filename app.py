# app.py — run with: streamlit run app.py

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="HAR — Activity Recogniser", page_icon="🏃", layout="centered")

@st.cache_resource
def load_artifacts():
    model         = joblib.load("har_model.pkl")
    scaler        = joblib.load("scaler.pkl")
    le            = joblib.load("label_encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")
    explainer     = shap.TreeExplainer(model)
    return model, scaler, le, feature_names, explainer

model, scaler, le, feature_names, explainer = load_artifacts()

EMOJI = {
    "LAYING": "🛌", "SITTING": "🪑", "STANDING": "🧍",
    "WALKING": "🚶", "WALKING_UPSTAIRS": "⬆️", "WALKING_DOWNSTAIRS": "⬇️"
}

st.sidebar.header("ℹ️ Model Info")
st.sidebar.markdown(f"""
- **Model:** {type(model).__name__}
- **Features:** {len(feature_names)}
- **Activities:** {len(le.classes_)}
- **Dataset:** UCI HAR
""")

st.title("🏃 Human Activity Recognition")
st.markdown(
    "Upload a CSV of sensor readings (561 features per row). "
    "The model predicts the activity and explains **why** using SHAP."
)
st.divider()

uploaded = st.file_uploader("Upload CSV (561 feature columns)", type="csv")

if uploaded:
    raw = pd.read_csv(uploaded)
    actual_labels = raw['Activity'].values if 'Activity' in raw.columns else None
    raw = raw.drop(columns=[c for c in ['Activity', 'subject'] if c in raw.columns])

    if raw.shape[1] != len(feature_names):
        st.error(f"Expected {len(feature_names)} columns, got {raw.shape[1]}.")
        st.stop()

    X_scaled = scaler.transform(raw.values)
    preds    = model.predict(X_scaled)
    labels   = le.inverse_transform(preds)

    st.success(f"{len(labels)} row(s) predicted.")
    st.divider()

    for i, label in enumerate(labels):
        actual_str = f" | Actual: `{actual_labels[i]}`" if actual_labels is not None else ""
        st.markdown(f"### Row {i+1}: {EMOJI.get(label,'❓')} `{label}`{actual_str}")

        proba    = model.predict_proba(X_scaled[i:i+1])[0]
        proba_df = pd.DataFrame({'Activity': le.classes_, 'Confidence': proba}).sort_values('Confidence')
        fig, ax  = plt.subplots(figsize=(7, 3))
        colors   = ['#1976D2' if a == label else '#B0BEC5' for a in proba_df['Activity']]
        bars     = ax.barh(proba_df['Activity'], proba_df['Confidence'], color=colors)
        ax.set_xlim(0, 1); ax.set_xlabel('Confidence'); ax.set_title('Prediction Confidence')
        for bar, val in zip(bars, proba_df['Confidence']):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.1%}', va='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        with st.expander(f"🔍 Why `{label}`? (SHAP explanation)"):
            sv = explainer.shap_values(X_scaled[i:i+1])
            shap.waterfall_plot(
                shap.Explanation(
                    values=sv[preds[i]][0],
                    base_values=explainer.expected_value[preds[i]],
                    data=X_scaled[i],
                    feature_names=feature_names
                ), max_display=12, show=False
            )
            plt.tight_layout(); st.pyplot(plt.gcf()); plt.close()
        st.divider()

    if len(labels) > 1:
        from collections import Counter
        counts = Counter(labels)
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.bar([EMOJI.get(k,'')+' '+k for k in counts], counts.values(),
                color='steelblue', edgecolor='black')
        ax2.set_title('Predicted Activity Counts'); ax2.set_ylabel('Count')
        plt.xticks(rotation=25, ha='right'); plt.tight_layout(); st.pyplot(fig2); plt.close()
        if actual_labels is not None:
            st.metric("Accuracy on uploaded file", f"{np.mean(labels==actual_labels):.1%}")

st.caption("Built with scikit-learn · SHAP · UCI HAR Dataset · Streamlit")
