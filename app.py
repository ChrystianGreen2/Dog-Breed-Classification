import os
import streamlit as st
from PIL import Image
from model import predict_dogs
import numpy as np

def load_image(image_file):
	img = Image.open(image_file)
	return img

if __name__ == '__main__':
    uploaded_file = st.file_uploader("Escolha um arquivo")
    if uploaded_file is not None:
        img = load_image(uploaded_file)
        st.title(str(predict_dogs(np.array(img))))
        st.image(img, caption=str(predict_dogs(np.array(img))))