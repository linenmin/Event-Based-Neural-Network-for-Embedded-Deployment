'''
Date: 2024-03-12
LastEditors: Claude Assistant
LastEditTime: 2024-03-12
FilePath: /Vitis-AI/pruning_tf2_v2.py
Description: Implementation of structured iterative pruning based on Vitis Optimizer TensorFlow2
'''
import os  # Import OS module for file path operations
import re  # Import regex module for extracting labels from filenames
import shutil  # Import file operations module for copying files
import numpy as np  # Import numpy for array operations
import tensorflow as tf  # Import tensorflow for deep learning
from tensorflow import keras  # Import keras for model building
from tf_nndct import IterativePruningRunner  # Import Vitis AI pruning tool
import glob  # Import glob module for file path matching
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show warnings and errors
tf.get_logger().setLevel('ERROR')

# Normalize event grid data
def normalize_event_grid(event_grid):
    """Normalize event grid data to [0,1] range"""
    min_vals = tf.reduce_min(event_grid, axis=[0, 1], keepdims=True)  # Calculate min value for each channel
    max_vals = tf.reduce_max(event_grid, axis=[0, 1], keepdims=True)  # Calculate max value for each channel
    normalized_grid = (event_grid - min_vals) / (max_vals - min_vals + 1e-8)  # Normalize, avoid division by zero
    return normalized_grid  # Return normalized data

# Parse TFRecord file
def parse_tfrecord_function(record, filename, class_to_label, num_classes):
    """Parse single TFRecord record and extract label from file path"""
    features = {
        'event_grid': tf.io.FixedLenFeature([], tf.string),  # Event grid data
        'shape': tf.io.FixedLenFeature([3], tf.int64)  # Data shape
    }
    example = tf.io.parse_single_example(record, features)  # Parse single sample
    event_grid = tf.io.decode_raw(example['event_grid'], tf.float32)  # Decode event grid data
    shape = example['shape']  # Get shape information
    event_grid = tf.reshape(event_grid, shape)  # Reshape data
    event_grid = tf.image.resize(event_grid, (224, 224))  # Resize to 224x224
    event_grid = normalize_event_grid(event_grid)  # Normalize data
    
    # Extract class name from file path (second-to-last part)
    parts = tf.strings.split(filename, os.sep)  # Split file path
    label_str = parts[-2]  # Get class name
    
    # Convert class name to numeric label
    label_int = tf.py_function(
        lambda x: class_to_label[x.numpy().decode('utf-8')], 
        [label_str], 
        tf.int32
    )  # Get numeric label through mapping
    label_int = tf.cast(label_int, tf.int32)  # Ensure correct type
    label_int.set_shape([])
    return event_grid, label_int  # Return data and label

# Create tf.data.Dataset from TFRecord file paths
def create_dataset_from_files(file_paths, labels, class_to_label, batch_size=32, shuffle=True):
    """Create TensorFlow dataset from TFRecord file paths"""
    num_classes = len(class_to_label)
    # Define function to process single file
    def process_file(filename):
        """Process single TFRecord file"""
        ds = tf.data.TFRecordDataset(filename)  # Create TFRecord dataset
        ds = ds.map(lambda record: (record, filename))  # Package record and filename
        return ds  # Return dataset
    # Create file path dataset
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)  # Create dataset from file paths
    # Use interleave to process multiple files
    ds = path_ds.interleave(
        lambda x: process_file(x),
        cycle_length=tf.data.AUTOTUNE,  # Auto-adjust cycle length
        block_length=1  # Process one block at a time
    )
    # Parse TFRecord data
    ds = ds.map(
        lambda record, filename: parse_tfrecord_function(record, filename, class_to_label, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE  # Parallel processing
    )
    if shuffle:  # If shuffling is needed
        ds = ds.shuffle(buffer_size=1000)  # Set buffer size
    ds = ds.batch(batch_size)  # Set batch size
    ds = ds.prefetch(tf.data.AUTOTUNE)  # Prefetch data for better performance
    return ds  # Return dataset

# Evaluate model accuracy
def evaluate_model(model, ds, steps=None):
    return model.evaluate(ds, steps=steps, verbose=0)[1]  # Evaluate model and return accuracy

# Add helper function to calculate model size
def get_model_size(model):
    """Calculate model size (MB)
    Args:
        model: TensorFlow model
    Returns:
        size_mb: Model size in MB
    """
    size = 0  # Initialize size
    for layer in model.layers:  # Iterate through all layers
        for weight in layer.weights:  # Iterate through layer weights
            size += np.prod(weight.shape) * 4  # Calculate weight size (assuming float32, 4 bytes)
    return size / (1024 * 1024)  # Convert to MB

# Iterative pruning (prune from scratch each round)
def iterative_pruning(model, train_ds, val_ds, ratios=[0.05, 0.1], out_dir='./prune_results', use_iterative=True):
    """Execute model pruning, supporting both iterative pruning and from-scratch pruning modes
    Args:
        model: Original model
        train_ds: Training dataset
        val_ds: Validation dataset
        ratios: List of pruning ratios, in ascending order
        out_dir: Output directory
        use_iterative: Whether to use iterative pruning (True: continue pruning based on previous round, False: start from original model each round)
    Returns:
        best_model: Best pruned model
        results: Pruning results record
    """
    input_spec = tf.TensorSpec((1,224,224,3), tf.float32)  # Define input specification
    best_model = None  # Best model
    best_acc = 0  # Best accuracy
    best_ratio = 0  # Best pruning ratio
    last_acc = evaluate_model(model, val_ds)  # Get original model accuracy
    
    orig_size = get_model_size(model)  # Calculate original model size
    orig_weights = model.get_weights()  # Save original weights for non-iterative mode
    current_model = model  # Current working model
    
    results = {  # Results record dictionary
        'original': {
            'acc': last_acc,  # Original accuracy
            'size': orig_size  # Original model size
        },
        'pruned': {}  # Pruning results
    }

    # Create results file and write original model information
    os.makedirs(out_dir, exist_ok=True)  # Create output directory
    with open(os.path.join(out_dir, 'detailed_results.txt'), 'w') as f:
        f.write("=== Original Model ===\n")
        f.write(f"Accuracy: {results['original']['acc']*100:.2f}%\n")
        f.write(f"Size: {results['original']['size']:.2f}MB\n\n")
        f.write(f"=== Pruned Models (Mode: {'Iterative' if use_iterative else 'From Scratch'}) ===\n")

    # Prune step by step according to ratios
    for i, target_ratio in enumerate(ratios):
        print(f"\nPruning iteration {i+1}/{len(ratios)}, target ratio: {target_ratio}")
        
        if not use_iterative:  # If not using iterative pruning, start from original model each round
            current_model = tf.keras.models.clone_model(model)  # Clone original model
            current_model.set_weights(orig_weights)  # Reset to original weights
            
        # Create pruning runner
        runner = IterativePruningRunner(current_model, input_spec)
        
        # Define model evaluation function
        small_ds = train_ds.shuffle(100).take(20)
        def compile_and_evaluate(m):
            m.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
            return evaluate_model(m, small_ds)
            
        # Analyze and prune
        runner.ana(compile_and_evaluate)
        sparse_model = runner.prune(ratio=target_ratio)
        
        # Calculate current model size
        current_size = get_model_size(sparse_model)
        
        # Fine-tuning parameters
        lr = 0.003 * (1 - target_ratio)  # Adjust learning rate based on pruning ratio
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        sparse_model.compile(optimizer, 'sparse_categorical_crossentropy', ['accuracy'])
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=10, 
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Fine-tuning training
        epochs = 100
        sparse_model.fit(
            train_ds, 
            epochs=epochs, 
            validation_data=val_ds, 
            callbacks=callbacks
        )
        
        # Evaluate current model
        current_acc = evaluate_model(sparse_model, val_ds)
        print(f"Current accuracy at ratio {target_ratio}: {current_acc*100:.2f}%")
        
        # Get and save current round's dense model
        slim = runner.get_slim_model()  # Get current round's dense model
        
        # Save current round's model
        weight_dir = os.path.join(out_dir, f'prune_{target_ratio}')  # Create save directory
        os.makedirs(weight_dir, exist_ok=True)  # Ensure directory exists
        
        # Save sparse model weights
        sparse_weights_path = os.path.join(weight_dir, 'sparse_weights')  # Sparse model weights path
        sparse_model.save_weights(sparse_weights_path, save_format='tf')  # Save sparse weights
        
        # Save dense model
        dense_model_path = os.path.join(weight_dir, 'dense_model')  # Dense model path
        slim.save(dense_model_path, save_format='tf')  # Save dense model
        
        # Record results
        results['pruned'][target_ratio] = {
            'acc': current_acc,
            'size': current_size,
            'sparse_weights': sparse_weights_path,  # Record sparse weights path
            'dense_model': dense_model_path     # Record dense model path
        }
        
        # Update results file
        with open(os.path.join(out_dir, 'detailed_results.txt'), 'a') as f:
            f.write(f"\nPrune ratio {target_ratio}:\n")
            f.write(f"Accuracy: {current_acc*100:.2f}%\n")
            f.write(f"Size: {current_size:.2f}MB\n")
            f.write(f"Sparse weights saved at: {sparse_weights_path}\n")
            f.write(f"Dense model saved at: {dense_model_path}\n")
        
        # Update best model
        if current_acc > best_acc:
            best_acc = current_acc
            best_model = slim
            best_ratio = target_ratio
            # Save best model
            best_model_dir = os.path.join(out_dir, 'best_model')
            os.makedirs(best_model_dir, exist_ok=True)
            # Save both sparse weights and dense model
            sparse_model.save_weights(os.path.join(best_model_dir, 'sparse_weights'), save_format='tf')
            best_model.save(os.path.join(best_model_dir, 'dense_model'), save_format='tf')
            # Record best model information
            with open(os.path.join(best_model_dir, 'info.txt'), 'w') as f:
                f.write(f"Best model found at ratio: {best_ratio}\n")
                f.write(f"Accuracy: {best_acc*100:.2f}%\n")
                f.write(f"Size: {get_model_size(best_model):.2f}MB\n")
            print(f"New best model found at ratio {target_ratio} with accuracy {best_acc*100:.2f}%")
        
        if use_iterative:  # If using iterative pruning, update current model
            current_model = slim

    print(f"\nBest model found at ratio {best_ratio} with accuracy {best_acc*100:.2f}%")
    
    # Save final experiment results
    results_file = os.path.join(out_dir, 'experiment_results.txt')
    with open(results_file, 'w') as f:
        f.write("=== Pruning Experiment Results ===\n\n")
        f.write(f"Pruning Mode: {'Iterative' if use_iterative else 'From Scratch'}\n\n")
        f.write("Original Model:\n")
        f.write(f"Accuracy: {results['original']['acc']*100:.2f}%\n")
        f.write(f"Size: {results['original']['size']:.2f}MB\n\n")
        f.write("Pruned Models:\n")
        for ratio, data in results['pruned'].items():
            f.write(f"\nRatio {ratio}:\n")
            f.write(f"Accuracy: {data['acc']*100:.2f}%\n")
            f.write(f"Size: {data['size']:.2f}MB\n")
        f.write(f"\nBest Model:\n")
        f.write(f"Ratio: {best_ratio}\n")
        f.write(f"Accuracy: {best_acc*100:.2f}%\n")
        f.write(f"Size: {get_model_size(best_model):.2f}MB\n")
    
    # Save final best model (already dense model)
    best_model.save(os.path.join(out_dir, 'final_pruned'), save_format='tf')  # Save as SavedModel format
    best_model.save(os.path.join(out_dir, 'final_pruned.h5'))  # Save as h5 format
    
    return best_model, results

# Split training/validation file paths (based on TFRecord files)
def split_file_paths(val_file_paths, data_dir, class_to_label):
    """Split training and validation file paths
    Args:
        val_file_paths: List of validation file paths
        data_dir: Dataset root directory
        class_to_label: Class name to label mapping (generated by main program)
    Returns:
        train_paths, train_labels, val_paths, val_labels
    """
    # Scan entire data directory for all TFRecord files
    all_paths = []  # Store all file paths
    for root, dirs, files in os.walk(data_dir):  # Recursively traverse directory
        for file in files:  # Iterate through each file
            if file.endswith('.tfrecord'):  # Only process TFRecord files
                all_paths.append(os.path.join(root, file))  # Add full path

    print(f"Found {len(class_to_label)} classes: {list(class_to_label.keys())}")  # Print class information
    val_set = set(val_file_paths)  # Convert to set for efficient lookup
    train_paths = []  # Training set paths
    for path in all_paths:  # Iterate through all file paths
        if path not in val_set:  # If not in validation set
            train_paths.append(path)  # Add to training set
    val_paths = val_file_paths  # Validation set paths are the input validation paths
    # Extract training set labels
    train_labels = []  # Training set labels
    for path in train_paths:  # Iterate through training set paths
        class_name = os.path.basename(os.path.dirname(path))  # Get class name
        train_labels.append(class_to_label[class_name])  # Map to numeric label
    # Extract validation set labels
    val_labels = []  # Validation set labels
    for path in val_paths:  # Iterate through validation set paths
        class_name = os.path.basename(os.path.dirname(path))  # Get class name
        val_labels.append(class_to_label[class_name])  # Map to numeric label
    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")  # Print dataset sizes
    return train_paths, train_labels, val_paths, val_labels  # Return split data

# Main program
if __name__ == '__main__':
    OUT_DIR = './prune_results_2'  # Unified output directory
    model_path = './models/8100Stack1281281_1of1_0.003_16_0.5_0.001_sgd_0.9_mbNetV2.h5'  # Model path
    DATA_DIR = './timeStack_1281281_1of1_tf'    
    VALID_DATA_TXT_FILE = './timeStack_1281281_1of1_tf_val_files.txt'
    ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # 1. Read file paths to process from text file
    if not os.path.exists(VALID_DATA_TXT_FILE): # Check if text file exists
        print(f"Error: Validation set file list {VALID_DATA_TXT_FILE} not found.")
    with open(VALID_DATA_TXT_FILE, 'r') as f: # Open text file for reading
        valid_files_relative = [line.strip() for line in f if line.strip()] # Read all non-empty lines
    valid_tfrecord_paths = [] # Initialize adjusted file paths list
    for rel_path in valid_files_relative: # Iterate through relative paths from txt
        path_parts = rel_path.replace("/", os.sep).split(os.sep) # Replace separators and split
        if len(path_parts) >= 2:
            class_name = path_parts[-2] # Class name
            file_name = path_parts[-1] # File name
            full_path = os.path.join(DATA_DIR, class_name, file_name)
            valid_tfrecord_paths.append(full_path) # Add to list
        else:
            print(f"Warning: Cannot extract class name and file name from path '{rel_path}', skipping this file.")
    if not valid_tfrecord_paths: # If no valid TFRecord paths
        print("No TFRecord file paths successfully parsed from text file.")
    print(f"Parsed {len(valid_tfrecord_paths)} target file paths from {VALID_DATA_TXT_FILE}.")
    # Unified construction of class_names and class_to_label
    all_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.tfrecord'):
                all_paths.append(os.path.join(root, file))
    class_names = sorted(list({os.path.basename(os.path.dirname(p)) for p in all_paths}))
    class_to_label = {name: i for i, name in enumerate(class_names)}
    # 2. Convert relative paths to absolute paths
    train_paths, train_labels, val_paths, val_labels = split_file_paths(valid_tfrecord_paths, DATA_DIR, class_to_label)  # Pass unified class_to_label
    train_ds = create_dataset_from_files(train_paths, train_labels, class_to_label, shuffle=True)  # Pass unified class_to_label
    val_ds = create_dataset_from_files(val_paths, val_labels, class_to_label, shuffle=False)  # Pass unified class_to_label

    # Model definition
    base = keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)  # Create MobileNetV2 base model
    x = keras.layers.GlobalAveragePooling2D()(base.output)  # Add global average pooling layer
    x = keras.layers.Dense(128, activation='relu')(x)  # Add fully connected layer
    out = keras.layers.Dense(10, activation='softmax')(x)  # Add output layer
    model = keras.Model(base.input, out)  # Create complete model
    model.load_weights(model_path)  # Load pre-trained weights
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])  # Compile model

    # Pre-pruning accuracy
    orig_acc = evaluate_model(model, val_ds)  # Evaluate original model accuracy
    print(f"Original acc: {orig_acc*100:.2f}%")  # Print original accuracy

    # # Iterative pruning mode
    # best_model, results = iterative_pruning(model, train_ds, val_ds, ratios, out_dir=OUT_DIR, use_iterative=True)

    # From-scratch pruning mode
    best_model, results = iterative_pruning(model, train_ds, val_ds, ratios, out_dir=OUT_DIR, use_iterative=False)

    # Recompile best model
    best_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Evaluate final model
    pruned_acc = evaluate_model(best_model, val_ds)  # Evaluate pruned model accuracy
    print(f"Pruned acc: {pruned_acc*100:.2f}%")  # Print pruned accuracy

    # Calculate final model metrics
    final_size = get_model_size(best_model)  # Get final model size

    # Write final model information
    with open(os.path.join(OUT_DIR, 'detailed_results.txt'), 'a') as f:
        f.write("\n=== Final Model ===\n")
        f.write(f"Accuracy: {pruned_acc*100:.2f}%\n")
        f.write(f"Size: {final_size:.2f}MB\n")

    print("Done.")  # Print completion message
