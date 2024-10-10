import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

# Load an image from a file, resize it, and convert it to a float32 tensor
def load_img(path_to_img):
    max_dim = 512
    img = PIL.Image.open(path_to_img)
    img = np.array(img)
    long_dim = max(img.shape[:2])
    scale = max_dim / long_dim
    img = tf.image.resize(img, (round(img.shape[0] * scale), round(img.shape[1] * scale)))
    img = img[tf.newaxis, :]
    return img

# Display an image using Matplotlib
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Using a pretrained model from TensorFlow Hub for Style Transfer
def apply_style_transfer(content_image_path, style_image_path, output_image_path):
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)
    
    # Load the model from TensorFlow Hub
    stylize = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    stylized_image = stylize(tf.constant(content_image), tf.constant(style_image))[0]
    
    # Save the stylized image
    output_image = tensor_to_image(stylized_image)
    output_image.save(output_image_path)
    print(f"Stylized image saved as {output_image_path}")

# Example usage:
content_path =r'C:\Users\PRATHAMESH\Downloads\horse_image.jpg'
style_path = r'"C:\Users\PRATHAMESH\Downloads\1-style.jpg'
output_path = r'C:\Users\PRATHAMESH\Downloads'

apply_style_transfer(content_path, style_path, output_path)
