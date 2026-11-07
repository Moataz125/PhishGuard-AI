
import streamlit as st
import pandas as pd
import base64
from pathlib import Path

from model import train_model, analyze_email


# =====================================
# 1. PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).parent

# Load CSS
css_file = BASE / "style.css"

if css_file.exists():
    st.html(
        f"<style>{css_file.read_text(encoding='utf-8')}</style>"
    )


# =====================================
# 2. LOAD MACHINE LEARNING MODEL
# =====================================

@st.cache_resource(
    show_spinner="Training Machine Learning Model..."
)
def load_model(path, modified_time):
    return train_model(path)


dataset = BASE.parent / "csvdata" / "Phishing_Email.csv "

if not dataset.exists():
    st.error(f"Dataset not found: {dataset}")
    st.stop()

try:
    model, metrics, matrix, total = load_model(
        str(dataset),
        dataset.stat().st_mtime_ns
    )
except Exception as error:
    st.error(f"Training failed: {error}")
    st.stop()


# =====================================
# 3. PROFILE IMAGE
# =====================================

photo = None

for filename in ["profile.jpg", "profile.png"]:
    candidate = BASE / filename
    if candidate.exists():
        photo = candidate
        break

if photo:
    image_data = base64.b64encode(
        photo.read_bytes()
    ).decode("utf-8")

    image_type = (
        "image/png" if photo.suffix == ".png"
        else "image/jpeg"
    )

    profile_html = (
        f'<img class="profile-photo" '
        f'src="data:{image_type};base64,{image_data}" '
        f'alt="Moataz Al-Enezi">'
    )
else:
    profile_html = '<div class="profile-placeholder">👨‍💻</div>'


# =====================================
# 4. SIDEBAR
# =====================================

with st.sidebar:

    st.title("🛡️ PhishGuard AI")
    st.write("Machine Learning Security Project")

    st.divider()

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Feature Extraction:** TF-IDF")
    st.write("**Language:** Python")
    st.write(f"**Dataset:** {total:,} emails")

    if hasattr(model, "threshold_"):
        st.write(f"**Threshold:** {model.threshold_:.2f}")

    st.divider()

    st.link_button(
        "LinkedIn Profile",
        "https://www.linkedin.com/in/moataz-farhan-60628830a/",
        use_container_width=True
    )

    st.link_button(
        "GitHub Profile",
        "https://github.com/Moataz125",
        use_container_width=True
    )

    st.divider()

    st.caption(
        "This tool analyzes email text only. "
        "It does not verify senders, URLs, or attachments."
    )


# =====================================
# 5. HEADER (ONLY ONCE)
# =====================================

st.html(f"""
<div class="hero">

    <div class="hero-top">
        <span class="brand">🛡️ PHISHGUARD AI</span>
        <span class="ml-badge">ML POWERED</span>
    </div>

    <h1>Intelligent Phishing Detection</h1>

    <p>
        Detect suspicious emails using Machine Learning
        and Natural Language Processing.
    </p>

    <div class="tags">
        <span>Python</span>
        <span>TF-IDF</span>
        <span>Logistic Regression</span>
    </div>

    <div class="developer-card">

        <div class="developer-icon">
            {profile_html}
        </div>

        <div class="developer-info">
            <small>DEVELOPED BY</small>
            <h3>Moataz Al-Enezi</h3>
            <p>Cybersecurity & Machine Learning</p>
        </div>

        <div class="social-links">

            <a class="linkedin-btn"
               href="https://www.linkedin.com/in/moataz-farhan-60628830a/"
               target="_blank"
               rel="noopener noreferrer">
                LinkedIn ↗
            </a>

            <a class="github-btn"
               href="https://github.com/Moataz125"
               target="_blank"
               rel="noopener noreferrer">
                GitHub ↗
            </a>

        </div>

    </div>

</div>
""")


# =====================================
# 6. MAIN TABS
# =====================================

tab1, tab2 = st.tabs([
    "🔍 Email Scanner",
    "📊 Model Performance"
])


# =====================================
# 7. EMAIL SCANNER
# =====================================

with tab1:

    st.subheader("Email Security Analysis")

    st.write(
        "Paste an email below to check whether "
        "it may contain phishing content."
    )

    email_text = st.text_area(
        "Email Content",
        height=220,
        placeholder="Paste email subject and body here..."
    )

    if st.button("🔍 Analyze Email", type="primary"):

        if not email_text.strip():
            st.warning("Please enter an email message.")

        else:
            prediction, probability = analyze_email(
                model, email_text
            )

            st.divider()
            st.subheader("Analysis Result")

            if prediction == 1:
                st.error("⚠️ Potential Phishing Email")
            else:
                st.success("✅ Likely Safe Email")

            st.metric(
                "Estimated Phishing Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)

            st.caption(
                "This is a machine learning estimate, "
                "not a guarantee of email safety."
            )


# =====================================
# 8. MODEL PERFORMANCE
# =====================================

with tab2:

    st.subheader("Machine Learning Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['Precision']:.2%}")
    col3.metric("Recall", f"{metrics['Recall']:.2%}")
    col4.metric("F1 Score", f"{metrics['F1 Score']:.2%}")

    st.divider()

    st.subheader("Confusion Matrix")

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Safe", "Actual Phishing"],
        columns=["Predicted Safe", "Predicted Phishing"]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    st.info(f"Total emails used: {total:,}")
