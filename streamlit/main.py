import streamlit as st

from PIL import Image

from utils import makePrediction

st.title("Card detection")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

captured_image = st.camera_input("Take picture of card(s)")

if uploaded_file or captured_image:
    if uploaded_file:
        image = uploaded_file
    else:
        image = Image.open(captured_image)

    result_image = makePrediction(image)

    st.image(
        result_image,
        caption="result image with bounding boxes",
        use_column_width=True,
    )

    st.text("To retry click 'x Clear photo'")
