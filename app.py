import os
import time
from io import BytesIO

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# إعداد الصفحة
st.set_page_config(
    page_title="X-ray Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Cairo', sans-serif;
        }

        .main-title {
            text-align: center;
            background: linear-gradient(135deg, #1e3a8a, #2563eb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #64748b;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .result-danger {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            border: 3px solid #dc2626;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-top: 15px;
        }

        .result-success {
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
            border: 3px solid #16a34a;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-top: 15px;
        }

        .result-title {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .danger-text {
            color: #991b1b;
        }

        .success-text {
            color: #166534;
        }

        .confidence {
            font-size: 22px;
            color: #1e293b;
            font-weight: 600;
        }

        .info-card {
            background: #f8fafc;
            border-left: 4px solid #2563eb;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<h1 class="main-title">🫁 COVID-19 X-ray Detection System</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">AI-Powered Chest X-ray Analysis</p>',
    unsafe_allow_html=True,
)


# تحميل نموذج TensorFlow Lite
MODEL_PATH = "covid_normal_model_small.tflite"


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found: {MODEL_PATH}. "
            "Make sure it is in the same folder as app.py."
        )
        st.stop()

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    return interpreter


with st.spinner("⏳ Loading AI model..."):
    interpreter = load_model()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# التنبؤ بالصورة
def predict_image(image):

    input_info = input_details[0]
    output_info = output_details[0]

    height = int(input_info["shape"][1])
    width = int(input_info["shape"][2])

    image = image.convert("RGB").resize((width, height))

    input_array = np.asarray(image, dtype=np.float32) / 255.0
    input_array = np.expand_dims(input_array, axis=0)

    input_dtype = input_info["dtype"]

    if np.issubdtype(input_dtype, np.integer):

        scale, zero_point = input_info["quantization"]

        if scale > 0:
            input_array = np.round(input_array / scale + zero_point)

        limits = np.iinfo(input_dtype)

        input_array = np.clip(
            input_array,
            limits.min,
            limits.max
        )

    input_array = input_array.astype(input_dtype)

    interpreter.set_tensor(
        input_info["index"],
        input_array
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_info["index"]
    )

    if np.issubdtype(output_info["dtype"], np.integer):

        scale, zero_point = output_info["quantization"]

        if scale > 0:
            output = (
                output.astype(np.float32) - zero_point
            ) * scale

    return float(np.asarray(output).reshape(-1)[0])


# واجهة التطبيق
col1, col2 = st.columns([1, 1])

with col1:

    st.markdown("### 📂 Upload X-ray Image")

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        st.image(
            image,
            caption=f"📷 {uploaded_file.name}",
            use_container_width=True
        )


with col2:

    st.markdown("### 🔬 Prediction Result")

    if uploaded_file is None:

        st.markdown(
            '<div class="info-card">'
            '👈 Please upload a chest X-ray image first'
            '</div>',
            unsafe_allow_html=True
        )

    elif st.button(
        "🔬 Analyze Image",
        use_container_width=True,
        type="primary"
    ):

        with st.spinner("⏳ Analyzing image..."):

            start_time = time.time()

            image = Image.open(
                BytesIO(uploaded_file.getvalue())
            ).convert("RGB")

            value = predict_image(image)

            elapsed = time.time() - start_time

        # تصنيف النتيجة حسب عتبة النموذج
        if value >= 0.5:

            result = "Normal"
            probability = value

            st.markdown(
                f"""
                <div class="result-success">
                    <div class="result-title success-text">
                        ✅ Normal
                    </div>
                    <div class="confidence">
                        Confidence: {probability * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            result = "COVID-19"
            probability = 1 - value

            st.markdown(
                f"""
                <div class="result-danger">
                    <div class="result-title danger-text">
                        ⚠️ COVID-19 Detected
                    </div>
                    <div class="confidence">
                        Confidence: {probability * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # عرض الاحتمالات
        st.markdown("### 📊 Probability Distribution")

        st.bar_chart(
            {
                "Probability (%)": [
                    value * 100,
                    (1 - value) * 100
                ]
            },
            height=250,
            use_container_width=True
        )

        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-around;">
                <div><b>Normal:</b> {value * 100:.2f}%</div>
                <div><b>COVID-19:</b> {(1-value) * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # التحليل التفصيلي
        with st.expander("📋 Detailed Analysis"):

            st.write(f"**Prediction:** {result}")

            st.write(
                f"**Raw Model Output:** {value:.4f}"
            )

            st.write(
                f"**Confidence:** {probability * 100:.2f}%"
            )

            st.write(
                f"**Processing Time:** {elapsed:.2f} seconds"
            )


# الشريط الجانبي
with st.sidebar:

    st.markdown("## 🫁 About")

    st.markdown(
        "**COVID-19 Detection System**\n\n"
        "AI-powered chest X-ray image classification "
        "using a CNN model."
    )

    st.markdown("---")

    st.markdown("### 📊 Model Info")

    st.markdown(
        "- **Input Size:** 224×224\n"
        "- **Classes:** Normal / COVID-19\n"
        "- **Framework:** TensorFlow Lite"
    )

    st.markdown("---")

    st.warning(
        "**⚠️ Disclaimer:** For educational and research "
        "purposes only. NOT a medical diagnosis. "
        "Always consult a qualified doctor."
    )


st.markdown("---")

st.markdown(
    """
    <p style='text-align:center; color:#94a3b8; font-size:13px;'>
        ⚠️ For educational and research purposes only — Not a medical diagnosis
        <br>
        Built with Streamlit + TensorFlow Lite
    </p>
    """,
    unsafe_allow_html=True
)