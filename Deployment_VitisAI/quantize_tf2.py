'''
Date: 2024-03-12 18:06:11
LastEditors: Zijie Ning zijie.ning@kuleuven.be
LastEditTime: 2024-05-06 14:02:43
FilePath: /Vitis-AI/tf2_quant.py
'''

import tempfile
import os

import tensorflow as tf
import numpy as np
import datetime
import cv2

from tensorflow import keras
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.layers import Conv2D as _Conv2D
from tensorflow.keras.layers import BatchNormalization as _BatchNormalization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# from keras.mixed_precision import Policy as DTypePolicy

# # Load ImageNet2012 dataset using TensorFlow Datasets
# dataset_builder = tfds.builder('imagenet2012_subset')
# dataset_builder.download_and_prepare()
# dataset_info = dataset_builder.info

# # Get training, validation, and test datasets
# train_dataset = dataset_builder.as_dataset(split='train')
# val_dataset = dataset_builder.as_dataset(split='validation')
# test_dataset = dataset_builder.as_dataset(split='test')

# # Print information about the dataset
# print("Number of classes:", dataset_info.features['label'].num_classes)
# print("Class names:", dataset_info.features['label'].names)
# print("Total number of training examples:", dataset_info.splits['train'].num_examples)
# print("Total number of validation examples:", dataset_info.splits['validation'].num_examples)
# print("Total number of test examples:", dataset_info.splits['test'].num_examples)


# --- Calibration Dataset Preparation ---
calib_data_dir = './valid_timeStack_1281281_1of1_tf_from_txt'  # Path to calibration images
val_data_dir = './valid_timeStack_1281281_1of1_tf_from_txt'    # Path to validation images
float_model_path = './models/0.9_tfmot_pruned.h5'
# --- Save quantized model ---
quantized_model_save_path = './quantize_result_tf2/0.9_tfmot_pruned_qat.h5'  # Suggested to include size info in filename

# Get all PNG files in the directory using tf.data.Dataset.list_files
calib_files = tf.data.Dataset.list_files(os.path.join(calib_data_dir, '*.png'), shuffle=False)
val_files = tf.data.Dataset.list_files(os.path.join(val_data_dir, '*.png'), shuffle=False)

# Function to load and preprocess images for calibration
def load_image(filename):
    # Read image file
    img = tf.io.read_file(filename)
    # Decode as RGB image
    img = tf.image.decode_png(img, channels=3)
    
    # Convert to float32
    img = tf.cast(img, tf.float32)
    # Resize image to 224x224
    img = tf.image.resize(img, [224, 224])
    
    # Calculate min and max values for each channel
    min_vals = tf.reduce_min(img, axis=[0, 1], keepdims=True)  # Calculate over height and width dimensions
    max_vals = tf.reduce_max(img, axis=[0, 1], keepdims=True)  # Calculate over height and width dimensions
    # Normalize each channel
    img = (img - min_vals) / (max_vals - min_vals + 1e-8)
    
    return img, 0

# Build calibration dataset
calib_ds = calib_files.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
calib_ds = calib_ds.batch(10).prefetch(tf.data.AUTOTUNE)

# Build validation dataset
val_ds = val_files.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(10).prefetch(tf.data.AUTOTUNE)




# train_images = tfds.load('imagenette', split='train[:100]', shuffle_files=True)
# train_images = train_images.shuffle(1000).batch(1).prefetch(100)
# # train_images = tf.image.resize(train_images, (224, 224))  # Resize the image to match ResNet50 input size


# --- Load pre-trained floating-point model ---
# Note: Load your high-accuracy H5 model that has been verified in test_h5model.ipynb
# This model should accept 224x224 input

input_tensor = tf.keras.Input(shape=(224, 224, 3))
base_model = tf.keras.applications.MobileNetV2(
    input_tensor=input_tensor,
    include_top=False,
    weights=None  # Can load pre-trained weights if available; otherwise random initialization
)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model_to_quantize = tf.keras.Model(inputs=input_tensor, outputs=outputs)

# Load weights if you have a trained model:
model_to_quantize.load_weights(float_model_path)

# model_to_quantize = tf.keras.models.load_model(float_model_path)
# model_to_quantize.summary() # 打印模型结构，确认其输入是 (None, 224, 224, 3)

# Function to evaluate model performance
def evaluate_model(model, val_files, steps=10):
    # Build validation dataset
    val_ds = val_files.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(10).prefetch(tf.data.AUTOTUNE)
    
    # Evaluate model
    loss, accuracy = model.evaluate(val_ds, steps=steps, verbose=0)
    
    # Explicitly clean up dataset
    del val_ds
    tf.keras.backend.clear_session()  # Clear Keras session cache
    
    return loss, accuracy

# --- Evaluate floating-point model performance ---
print("\n--- Evaluating Floating-point Model Performance ---")
model_to_quantize.compile(
    optimizer='adam',  # Specify optimizer (adam is usually a good default)
    loss='sparse_categorical_crossentropy',  # Specify loss function (suitable for integer labels and softmax output)
    metrics=['accuracy']  # Specify evaluation metric (accuracy)
)
# Evaluate floating-point model and get results
float_loss, float_accuracy = evaluate_model(model_to_quantize, val_files)
print(f"Floating-point Model - Loss: {float_loss:.4f}")
print(f"Floating-point Model - Accuracy: {float_accuracy*100:.2f}%")

# --- Perform Quantization ---
print("\nStarting model quantization...")
# Create quantizer with the model to be quantized
quantizer = vitis_quantize.VitisQuantizer(model_to_quantize)

calib_ds = calib_ds.take(100)
# Quantize with QAT fine-tuning
quantized_model = quantizer.quantize_model(
    calib_dataset=calib_ds,
    calib_steps=100,  # For example, if calib_ds has 10 batches (100 images/batch_size=10), set to 10
    calib_batch_size=1,
    include_fast_ft=True, fast_ft_epochs=5
)

# --- Evaluate quantized model performance ---
print("\n--- Evaluating Quantized Model Performance ---")
quantized_model.compile(
    optimizer='adam',  # Specify optimizer
    loss='sparse_categorical_crossentropy',  # Specify loss function
    metrics=['accuracy']  # Specify evaluation metric
)
# Evaluate quantized model and get results
quantized_loss, quantized_accuracy = evaluate_model(quantized_model, val_files)
print(f"Quantized Model - Loss: {quantized_loss:.4f}")
print(f"Quantized Model - Accuracy: {quantized_accuracy*100:.2f}%")

# --- Print performance comparison ---
print("\n--- Performance Comparison Before and After Quantization ---")
print(f"Loss change: from {float_loss:.4f} (float) to {quantized_loss:.4f} (quantized)")
accuracy_drop = (float_accuracy - quantized_accuracy) * 100  # Calculate accuracy drop percentage
print(f"Accuracy change: from {float_accuracy*100:.2f}% (float) to {quantized_accuracy*100:.2f}% (quantized) (drop: {accuracy_drop:.2f}%)")


# --- Save quantized model ---

print(f"Quantization complete, saving quantized model to: {quantized_model_save_path}")
quantized_model.save(quantized_model_save_path)
print("Quantized model saved.")


# model = tf.keras.applications.resnet50.ResNet50(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000
# )

# model = tf.keras.applications.mobilenet_v2.MobileNetV2(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000
# )
# calib_images = train_images.map(lambda x: (x['image'], x['label']))

# from tensorflow_model_optimization.quantization.keras import vitis_quantize
# quantizer = vitis_quantize.VitisQuantizer(model)
# quantized_model = quantizer.quantize_model(calib_dataset=calib_images, 
#                                            calib_steps=100, 
#                                            calib_batch_size=10) 
# quantized_model.save('resnet_quantized.h5')
# # quantized_model.save('mobilenetV2_quantized.h5')



