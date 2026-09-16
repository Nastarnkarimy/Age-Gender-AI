import streamlit as st 
import tensorflow as tf 
import numpy as np 
from PIL import Image 



st.set_page_config( 
    page_title="Age & Gender AI", 
    page_icon="🧠", 
    layout="wide" 
) 


st.markdown(""" 
<style> 

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 45%,
        #1e293b 100%
    );
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #f9a8d4;
    margin-bottom: 40px;
}

.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
}

.result-title {
    color: #f9a8d4;
    font-size: 15px;
}

.result-value {
    color: white;
    font-size: 32px;
    font-weight: 700;
}

.model-info {
    text-align: center;
    color: #f9a8d4;
    margin-top: 30px;
}


/* =========================
   PINK UPLOAD BUTTON
   ========================= */

[data-testid="stFileUploader"] button {
    background-color: #ec4899 !important;
    color: white !important;
    border: 1px solid #f472b6 !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #db2777 !important;
    color: white !important;
}

[data-testid="stFileUploader"] section {
    background: rgba(236, 72, 153, 0.06) !important;
    border-color: rgba(244, 114, 182, 0.6) !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: #ec4899 !important;
}

[data-testid="stFileUploader"] svg {
    color: #f472b6 !important;
}


/* =========================
   PINK PROGRESS BAR
   ========================= */

[data-testid="stProgress"] > div > div > div > div {
    background-color: #ec4899 !important;
}


/* =========================
   PINK FOCUS / ACCENT
   ========================= */

input:focus,
textarea:focus {
    border-color: #ec4899 !important;
    box-shadow: 0 0 0 1px #ec4899 !important;
}

</style> 
""", unsafe_allow_html=True)




# =========================
# LOAD MODEL
# =========================


@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "age_gender_mobilenetv2.keras"
    )

    return model



model = load_model()





# =========================
# PREPROCESS
# =========================


def preprocess_image(image):

    image = image.copy()

    image = image.convert("RGB")


    image = image.resize(
        (128,128),
        Image.Resampling.LANCZOS
    )


    image = np.array(image)


    image = image.astype(
        np.float32
    ) / 255.0


    image = np.expand_dims(
        image,
        axis=0
    )


    return image





# =========================
# PREDICT
# =========================


def predict(image):

    image_array = preprocess_image(
        image
    )


    prediction = model.predict(
        image_array,
        verbose=0
    )


    # طبق تست خودت:
    # prediction[0] -> gender
    # prediction[1] -> age


    gender_raw = float(
        prediction[0][0][0]
    )


    age_raw = float(
        prediction[1][0][0]
    )



    if gender_raw >= 0.5:

        gender = "Female"

        confidence = gender_raw


    else:

        gender = "Male"

        confidence = 1 - gender_raw



    age = int(
        np.clip(
            round(age_raw),
            0,
            116
        )
    )



    confidence_percent = round(
        confidence * 100,
        1
    )


    return (
        age,
        gender,
        confidence,
        confidence_percent,
        gender_raw,
        age_raw
    )






# =========================
# HEADER
# =========================


st.markdown(
    '<div class="main-title">Age & Gender AI</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">'
    'Multi Task CNN for Age Estimation and Gender Classification'
    '</div>',
    unsafe_allow_html=True
)





left, right = st.columns(
    [1,1],
    gap="large"
)





# =========================
# LEFT
# =========================


with left:


    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )


    st.subheader(
        "Upload Image"
    )


    uploaded_file = st.file_uploader(
        "Choose face image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )



    if uploaded_file is not None:


        image = Image.open(
            uploaded_file
        ).copy()



        st.image(
            image,
            width=400
        )






# =========================
# RIGHT
# =========================


with right:


    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )


    st.subheader(
        "AI Prediction"
    )



    if uploaded_file is None:


        st.info(
            "Upload an image"
        )



    else:


        with st.spinner(
            "Analyzing..."
        ):


            (
                age,
                gender,
                confidence,
                confidence_percent,
                gender_raw,
                age_raw

            ) = predict(image)




        col1, col2 = st.columns(2)




        with col1:


            st.markdown(
                '<div class="result-title">Age</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                f'<div class="result-value">{age} years</div>',
                unsafe_allow_html=True
            )




        with col2:


            st.markdown(
                '<div class="result-title">Gender</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                f'<div class="result-value">{gender}</div>',
                unsafe_allow_html=True
            )






        st.markdown(
            '<div class="result-title">'
            'Gender Confidence'
            '</div>',
            unsafe_allow_html=True
        )



        st.progress(
            float(confidence)
        )



        st.caption(
            f"Confidence: {confidence_percent}%"
        )



        # برای بررسی فقط در صورت نیاز
        # بعداً می‌توانی حذف کنی

        with st.expander("Debug"):

            st.write(
                "Raw gender:",
                gender_raw
            )

            st.write(
                "Raw age:",
                age_raw
            )



    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )





# =========================
# FOOTER
# =========================


st.markdown(
    '<div class="model-info">'
    'Built with TensorFlow • Keras • Streamlit'
    '</div>',
    unsafe_allow_html=True
)