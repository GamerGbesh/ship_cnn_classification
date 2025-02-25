import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def main():
    train_categories = {0 : 'Cargo',
                        1 : 'Military',
                        2 : 'Carrier',
                        3 : 'Cruise',
                        4 : 'Tankers'}
    target_size = (128, 128)
    st.title("Ship detector")
    file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])
    if file:
        img = tf.keras.utils.load_img(file, target_size=target_size)
        img = tf.image.convert_image_dtype(img, tf.float16)
        img = tf.keras.utils.img_to_array(img)
        st.image(img)

        model = tf.keras.models.load_model("ship_model.keras")
        prediction = np.argmax(model.predict(img.reshape(1, *target_size, 3)))
        prediction = train_categories[prediction]
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()

