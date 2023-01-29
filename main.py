import cv2
import numpy as np
import streamlit as st
from model import Yolo_v5
st.markdown("<h1 style='text-align: center; color: black;'>Detect CS GO persons.</h1>", unsafe_allow_html=True)

model = Yolo_v5()

if __name__ == "__main__":
    
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="BGR", caption='Source image')
        result = model.predict(img)
        result_img = result.render()
        
        st.image(result_img, channels="BGR", caption='Detected persons')
