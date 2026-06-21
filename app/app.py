import sys
import time
import streamlit as st

# Import predict functions & configuration from src
try:
    from src.predict import BertPredictor, MambaPredictor, build_input_text
except ImportError as e:
    st.error(
        f"Failed to import source files. Make sure you run the app from the project root directory.\n"
        f"Error: {e}"
    )
    st.stop()

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Bangla Fake News Detector",
    page_icon="🇧🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Preloaded Example Data ────────────────────────────────────
EXAMPLES = {
    "real_1": {
        "title": "আইটি ট্রেনিং সেন্টারের ভিত্তি স্থাপন করলেন রাষ্ট্রপতি",
        "body": "নেত্রকোনা: নেত্রকোনা শহরের পুরাতন জেলখানা সড়কে শেখ কামাল আইটি ট্রেনিং অ্যান্ড ইনকিউবেশন সেন্টারের ভিত্তিপ্রস্তর স্থাপন করেছেন রাষ্ট্রপতি মো. আবদুল হামিদ। বুধবার বিকেলে তিনি এ ভিত্তিপ্রস্তর স্থাপন করেন। এর আগে তিনি হেলিকপ্টারে করে নেত্রকোনা বর্ডার গার্ড বাংলাদেশ (বিজিবি) ক্যাম্পে এসে পৌঁছান। অনুষ্ঠানে রাষ্ট্রপতি বলেন, এ অঞ্চলের তরুণ সমাজকে দক্ষ মানবসম্পদ হিসেবে গড়ে তুলতে এই ট্রেনিং সেন্টার গুরুত্বপূর্ণ ভূমিকা রাখবে।"
    },
    "real_2": {
        "title": "বঙ্গবন্ধু স্যাটেলাইট-১ এর সফল উৎক্ষেপণ",
        "body": "মহাকাশে সফলভাবে উৎক্ষেপণ করা হয়েছে বাংলাদেশের প্রথম যোগাযোগ উপগ্রহ বঙ্গবন্ধু স্যাটেলাইট-১। যুক্তরাষ্ট্রের ফ্লোরিডার কেপ কেনাভেরাল থেকে স্পেসএক্সের ফ্যালকন ৯ রকেটের মাধ্যমে এটি উৎক্ষেপণ করা হয়। এর মাধ্যমে বিশ্বের ৫৭তম দেশ হিসেবে নিজস্ব স্যাটেলাইটের মালিক হলো বাংলাদেশ, যা দেশের গৌরব ও প্রযুক্তির অগ্রযাত্রাকে নির্দেশ করে।"
    },
    "fake_1": {
        "title": "পদ্মা সেতুতে মানুষের মাথা ও রক্ত লাগবে বলে গুজব",
        "body": "সামাজিক যোগাযোগ মাধ্যমে ছড়িয়ে পড়েছে যে পদ্মা সেতু তৈরিতে মানুষের মাথা ও রক্ত লাগবে। এই গুজবে কান দিয়ে দেশের বিভিন্ন স্থানে ছেলেধরা সন্দেহে গণপিটুনির ঘটনা ঘটছে। আইনশৃঙ্খলা রক্ষাকারী বাহিনী জানিয়েছে এটি সম্পূর্ণ ভিত্তিহীন ও সাজানো গুজব। সবাইকে সতর্ক থাকার অনুরোধ করা হলো এবং আইন নিজের হাতে না তুলে নেওয়ার আহ্বান জানানো হয়েছে।"
    },
    "fake_2": {
        "title": "করোনা নিরাময়ে থানকুনি পাতার অলৌকিক রস",
        "body": "করোনাভাইরাস প্রতিরোধে থানকুনি পাতার রস খেলে শতভাগ নিরাময় পাওয়া যায় বলে রাতে বিভিন্ন এলাকায় রটানো হয়েছে। খবরটি ছড়ানোর পর গভীর রাতে মানুষের মধ্যে পাতা সংগ্রহের হিড়িক পড়ে যায়। স্বাস্থ্য বিশেষজ্ঞরা জানিয়েছেন, এটি বৈজ্ঞানিকভাবে প্রমাণিত নয় এবং এটি সম্পূর্ণ মিথ্যা অপপ্রচার ও কুসংস্কার।"
    }
}

# Initialize session state for input fields
if "news_title" not in st.session_state:
    st.session_state.news_title = ""
if "news_body" not in st.session_state:
    st.session_state.news_body = ""

def load_example(example_key):
    st.session_state.news_title = EXAMPLES[example_key]["title"]
    st.session_state.news_body = EXAMPLES[example_key]["body"]

@st.cache_resource(show_spinner="Loading BanglaBERT model to RAM...")
def get_bert_predictor():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    try:
        predictor = BertPredictor(device="cpu")
        predictor.load()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    return predictor

@st.cache_resource(show_spinner="Loading Bangla-Mamba model to RAM...")
def get_mamba_predictor():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    try:
        predictor = MambaPredictor(device="cpu")
        predictor.load()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    return predictor

# ── Custom Styling (CSS Injection) ────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Noto+Sans+Bengali:wght@300;400;500;600;700&display=swap');
    
    /* Main layout fonts */
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Noto Sans Bengali', sans-serif;
    }
    
    /* Main header styling */
    .header-container {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .header-container h1 {
        color: white !important;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }
    .header-container p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Section headers */
    h2, h3 {
        font-weight: 700;
    }
    
    /* Sidebar styling tweaks */
    .css-1d391wt {
        padding-top: 2rem;
    }
    
    /* Examples section styling */
    .example-btn-container {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    
    /* Custom Result Cards */
    .result-card {
        padding: 1.75rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .real-card {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        border-left: 8px solid #0a5f57;
    }
    .fake-card {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        border-left: 8px solid #9c1c24;
    }
    .card-icon {
        font-size: 3.5rem;
    }
    .card-content h3 {
        margin: 0 0 0.5rem 0;
        color: white !important;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .card-content p {
        margin: 0;
        font-size: 1.05rem;
        opacity: 0.95;
    }
    
    /* Custom progress/probability meter */
    .meter-container {
        background: #f1f3f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    .meter-bar-wrapper {
        margin-bottom: 1rem;
    }
    .meter-label {
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }
    .meter-bg {
        background-color: #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
        height: 14px;
        width: 100%;
    }
    .meter-fill-real {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease-out;
    }
    .meter-fill-fake {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── App Title & Header ────────────────────────────────────────
st.markdown(
    """
    <div class="header-container">
        <h1>Bangla Fake News Detector</h1>
        <p>Fine-tuned Deep Learning and State-Space Models (SSM) for identifying disinformation and fake news in Bangla.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ── Sidebar Configurations ────────────────────────────────────
st.sidebar.title("🛠️ Configurations")
st.sidebar.markdown("---")

# 1. Model Selector
model_option = st.sidebar.selectbox(
    "Choose Prediction Model",
    ("BanglaBERT (Fine-tuned Transformer)", "Bangla-Mamba (State-Space Model)"),
    index=0,
    help="Select the deep learning architecture to run inference."
)
model_choice = "bert" if "BanglaBERT" in model_option else "mamba"

# 2. Execution Device
device = "cpu"

# 3. Model Information Panels
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Architectures")
if model_choice == "bert":
    st.sidebar.info(
        "**BanglaBERT:**\n"
        "A BERT-based model pretrained on 2.75 GB of Bangla web crawl corpora. "
        "Fine-tuned specifically for binary classification of real vs. fake news headlines and content."
    )
else:
    st.sidebar.info(
        "**Bangla-Mamba:**\n"
        "A State-Space Model (SSM) trained from scratch using the BanglaBERT tokenizer. "
        "Offers high speed, linear sequence scaling, and a larger context length (768 tokens vs 512 for BERT)."
    )

st.sidebar.markdown(f"**Selected Device:** `{device.upper()}`")

# ── Main Content Area ─────────────────────────────────────────

# A. Example Quick-Load Section
st.subheader("💡 Quick Test Examples")
st.markdown("Select one of the pre-loaded news articles to pre-fill the form fields below:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📰 Real News 1 (IT Center)", use_container_width=True):
        load_example("real_1")
with col2:
    if st.button("📰 Real News 2 (Satellite)", use_container_width=True):
        load_example("real_2")
with col3:
    if st.button("⚠️ Fake News 1 (Bridge)", use_container_width=True):
        load_example("fake_1")
with col4:
    if st.button("⚠️ Fake News 2 (Herb)", use_container_width=True):
        load_example("fake_2")

st.markdown("---")

# B. News Input Form
st.subheader("✍️ Analyze News Article")

# Bind inputs directly to session state
title_input = st.text_input(
    "সংবাদের শিরোনাম (Headline/Title)", 
    key="news_title",
    placeholder="সংবাদের মূল আকর্ষণ বা শিরোনাম এখানে লিখুন..."
)

body_input = st.text_area(
    "সংবাদের মূল বিষয়বস্তু (News Body/Content)",
    key="news_body",
    placeholder="পুরো খবর বা মূল বিষয়বস্তু এখানে কপি-পেস্ট করুন...",
    height=250
)

# Character and Word counters
words_count = len(body_input.split()) if body_input.strip() else 0
chars_count = len(body_input) if body_input.strip() else 0
st.markdown(
    f"<p style='text-align: right; color: gray; font-size: 0.85rem; margin-top: -10px;'>"
    f"Words: {words_count} | Characters: {chars_count}</p>", 
    unsafe_allow_html=True
)

predict_btn = st.button("🔍 সংবাদের সত্যতা যাচাই করুন (Verify News)", type="primary", use_container_width=True)

# C. Inference & Result Presentation
if predict_btn:
    if not title_input.strip() and not body_input.strip():
        st.warning("❌ অনুগ্রহ করে শিরোনাম অথবা সংবাদের মূল বিষয়বস্তু প্রদান করুন। (Please provide a title or body.)")
    else:
        # Load the selected model
        try:
            if model_choice == "bert":
                predictor = get_bert_predictor()
                backend = "BanglaBERT (HuggingFace)"
            else:
                predictor = get_mamba_predictor()
                backend = predictor.backend_name

            # Perform prediction
            with st.spinner("⏳ সংবাদটি যাচাই করা হচ্ছে, অনুগ্রহ করে অপেক্ষা করুন... (Analyzing news text...)"):
                start_time = time.time()
                
                # Apply preprocessing pipeline
                cleaned_text = build_input_text(title_input, body_input)
                
                # Perform inference
                result = predictor.predict(cleaned_text)
                
                latency = time.time() - start_time
                token_count = len(predictor.tokenizer.encode(cleaned_text, truncation=False))

            # Success response UI
            st.success("✅ বিশ্লেষণ সম্পন্ন হয়েছে! (Analysis Completed!)")

            # Get labels and values
            label = result["label"]            # "Fake 🔴" or "Real ✅"
            prob_fake = result["prob_fake"]
            prob_real = result["prob_real"]
            confidence = result["confidence"]
            
            is_fake = "Fake" in label

            # 1. Prediction Outcome Card
            if is_fake:
                st.markdown(
                    f"""
                    <div class="result-card fake-card">
                        <div class="card-icon">🚨</div>
                        <div class="card-content">
                            <h3>সন্দেহজনক খবর (FAKE NEWS DETECTED)</h3>
                            <p>সংবাদটি আমাদের এআই মডেল দ্বারা বিভ্রান্তিকর, অসত্য বা গুজব হিসেবে সনাক্ত করা হয়েছে। (Model classified this news as fake, misleading, or a rumor.)</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card real-card">
                        <div class="card-icon">✅</div>
                        <div class="card-content">
                            <h3>নির্ভরযোগ্য খবর (REAL NEWS DETECTED)</h3>
                            <p>সংবাদটি আমাদের এআই মডেল দ্বারা সত্য ও নির্ভরযোগ্য হিসেবে সনাক্ত করা হয়েছে। (Model classified this news as real, authentic, and reliable.)</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 2. Probability Meters
            st.subheader("📊 Probability Breakdown")
            st.markdown(
                f"""
                <div class="meter-container">
                    <div class="meter-bar-wrapper">
                        <div class="meter-label">
                            <span>সত্য সংবাদের সম্ভাবনা (Real Probability)</span>
                            <span>{prob_real * 100:.2f}%</span>
                        </div>
                        <div class="meter-bg">
                            <div class="meter-fill-real" style="width: {prob_real * 100}%;"></div>
                        </div>
                    </div>
                    <div class="meter-bar-wrapper" style="margin-bottom: 0;">
                        <div class="meter-label">
                            <span>ভুয়া সংবাদের সম্ভাবনা (Fake Probability)</span>
                            <span>{prob_fake * 100:.2f}%</span>
                        </div>
                        <div class="meter-bg">
                            <div class="meter-fill-fake" style="width: {prob_fake * 100}%;"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 3. Model & Technical Details Table
            st.subheader("⚙️ Technical Performance Metrics")
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            with col_metric1:
                st.metric("Model Architecture", "BanglaBERT" if model_choice == "bert" else "Bangla-Mamba")
            with col_metric2:
                st.metric("Inference Latency", f"{latency:.3f} seconds")
            with col_metric3:
                st.metric("Processed Tokens", f"{token_count}")
            with col_metric4:
                st.metric("Execution Backend", "CPU" if device == "cpu" else "GPU (CUDA)")

            # 4. Text Preprocessing Details
            with st.expander("🔍 Preprocessing and Input Text Analysis"):
                st.markdown("**Cleaned Input Text (merged title and body with `[SEP]` and normalized):**")
                st.info(cleaned_text)
                
                st.markdown("**RAW Input Information:**")
                st.json({
                    "raw_title": title_input,
                    "raw_body": body_input,
                    "cleaned_length_chars": len(cleaned_text),
                    "token_count": token_count,
                    "confidence_score": confidence,
                    "backend_engine": backend
                })

        except Exception as e:
            import traceback
            st.error(f"⚠️ একটি ত্রুটি ঘটেছে: {e}")
            st.code(traceback.format_exc(), language="python")
            st.info(
                "Verify that the pre-trained weights exist in the `Artifacts/best_model` folder.\n"
                "For BanglaBERT: `Artifacts/best_model/banglabert` must contain the PyTorch model files.\n"
                "For Bangla-Mamba on CPU: `Artifacts/best_model/mamba_768_hf` must contain the converted PyTorch weights."
            )
