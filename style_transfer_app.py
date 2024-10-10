import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import io

# Load an image from a file, resize it, and convert it to a float32 tensor
def load_img(image_file):
    max_dim = 512
    img = PIL.Image.open(image_file)
    img = np.array(img)
    long_dim = max(img.shape[:2])
    scale = max_dim / long_dim
    img = tf.image.resize(img, (round(img.shape[0] * scale), round(img.shape[1] * scale)))
    img = img[tf.newaxis, :]
    return img

# Convert tensor to an image and return PIL image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Using a pretrained model from TensorFlow Hub for Style Transfer
def apply_style_transfer(content_image, style_image):
    # Load the model from TensorFlow Hub
    stylize = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Stylize the image
    stylized_image = stylize(tf.constant(content_image), tf.constant(style_image))[0]
    return tensor_to_image(stylized_image)

# Streamlit app UI
st.title("Neural Style Transfer")

# Upload content and style images
content_image_file = st.file_uploader("Choose a content image...", type=["jpg", "png"])
style_image_file = st.file_uploader("Choose a style image...", type=["jpg", "png"])

# Display the uploaded images
if content_image_file and style_image_file:
    content_image = load_img(content_image_file)
    style_image = load_img(style_image_file)

    st.image([content_image_file, style_image_file], caption=["Content Image", "Style Image"], width=300)

    # Apply style transfer when the button is clicked
    if st.button("Stylize Image"):
        stylized_image = apply_style_transfer(content_image, style_image)

        # Display the stylized image
        st.image(stylized_image, caption="Stylized Image", use_column_width=True)

        # Create a button to download the image
        img_buffer = io.BytesIO()
        stylized_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        st.download_button(
            label="Download Stylized Image",
            data=img_buffer,
            file_name="stylized_image.jpg",
            mime="image/jpeg"
        )
