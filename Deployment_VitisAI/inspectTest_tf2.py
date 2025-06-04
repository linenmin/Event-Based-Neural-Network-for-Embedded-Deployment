import tempfile
import os
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

# === Your custom model definition ===
l2_reg = 0.001
dropout_rate = 0.5

input_tensor = Input(shape=(224, 224, 3))
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(x)
x = Dropout(dropout_rate)(x)

outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# === Load your trained weights ===
model.load_weights("./models/0.9_tfmot_pruned.h5")

# === Ensure output directory exists ===
output_dir = "inspect_tf2_mbNetV2"
os.makedirs(output_dir, exist_ok=True)

# === Vitis inspect check ===
from tensorflow_model_optimization.quantization.keras import vitis_inspect

inspector = vitis_inspect.VitisInspector(target="DPUCZDX8G_ISA1_B4096")

inspector.inspect_model(model, 
                        plot=True, 
                        plot_file=os.path.join(output_dir, "model.svg"), 
                        dump_results=True, 
                        dump_results_file=os.path.join(output_dir, "inspect_results.txt"), 
                        verbose=0)

print(f"Inspection results saved in: {output_dir}")
