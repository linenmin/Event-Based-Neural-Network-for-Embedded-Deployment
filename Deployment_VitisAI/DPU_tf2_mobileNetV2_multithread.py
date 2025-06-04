#!/usr/bin/env python3
"""
model_inference.py
Inference deployment script based on training code, used to run quantized xmodel on the board,
and calculate overall average accuracy (Top1) by extracting ground-truth labels from image filenames.

Usage: python3 model_inference.py <thread_number> <xmodel_file> <image_dir>
"""

import os
import sys
import time
import threading
import math
import re
import cv2
import numpy as np
import xir
import vart

#############################
# Global counters for correct predictions and total samples (thread-safe)
#############################
global_correct = 0
global_total = 0
counter_lock = threading.Lock()
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

#############################
# 1. Helper functions: Softmax and TopK
#############################
def CPUCalcSoftmax(data, size, scale):
    data = data.astype(np.float32)
    exp_data = np.exp(data * scale)
    sum_val = np.sum(exp_data)
    return (exp_data / sum_val).tolist()

def TopK(datain, k=5):
    datain = np.array(datain).flatten()  # Convert datain to 1D array
    indices = np.argsort(datain)[::-1][:k]
    values = [datain[int(i)] for i in indices]
    return indices, values

#############################
# 3. Extract ground-truth label from filename
#############################
def extract_label_from_filename(fname):
    """
    Assuming filename format: calib_classX_YY.png, returns X (integer type)
    """
    m = re.search(r"calib_class(\d+)_", fname)
    if m:
        return int(m.group(1))
    else:
        return None

#############################
# 4. Inference execution function (with accuracy statistics)
#############################
def runModel(runner, img_list, gt_list, total_iter):
    """
    runner: DPU Runner
    img_list: List of preprocessed images, each with shape (width, height, 3), dtype int8
    gt_list: Corresponding ground-truth label list
    total_iter: Total iterations for each thread (batch size is first dimension of input tensor)
    """
    global global_correct, global_total, counter_lock

    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    in_shape = tuple(inputTensors[0].dims)   # e.g. (batch_size, 128, 128, 3)
    out_shape = tuple(outputTensors[0].dims)   # e.g. (batch_size, 10)
    batch_size = in_shape[0]
    out_size = outputTensors[0].get_data_size() // batch_size
    out_fixpos = outputTensors[0].get_attr("fix_point")
    out_scale = 1.0 / (2 ** out_fixpos)

    n_images = len(img_list)
    count = 0
    while count < total_iter:
        inputData = [np.empty(in_shape, dtype=np.int8, order="C")]
        outputData = [np.empty(out_shape, dtype=np.int8, order="C")]
        for j in range(batch_size):
            idx = (count + j) % n_images
            inputData[0][j, ...] = img_list[idx]
        job_id = runner.execute_async(inputData, outputData)
        runner.wait(job_id)
        # Calculate softmax and count prediction accuracy for each sample
        for j in range(batch_size):
            softmax = CPUCalcSoftmax(outputData[0][j], out_size, out_scale)
            predicted = np.argmax(softmax)
            gt = gt_list[(count + j) % n_images]
            with counter_lock:
                global_total += 1
                if predicted == gt:
                    global_correct += 1
            # Only print TopK information for first batch (adjustable as needed)
            if count < batch_size:
                indices, values = TopK(softmax, k=5)
                print(f"Sample {count+j} - GT: {gt}, Predicted: {predicted}")
                for idx, val in zip(indices, values):
                    print(f"  Index: {idx}, Probability: {val:.4f}")
        count += batch_size

#############################
# 5. Get DPU subgraph function
#############################
def get_child_subgraph_dpu(graph):
    root_subgraph = graph.get_root_subgraph()
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [cs for cs in child_subgraphs if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"]

#############################
# 2. Preprocessing functions
#############################
def normalize_image(image):
    """Normalize input image to [0,1] using min-max normalization"""
    min_vals = np.min(image, axis=(0, 1), keepdims=True)
    max_vals = np.max(image, axis=(0, 1), keepdims=True)
    return (image - min_vals) / (max_vals - min_vals + 1e-8)

def preprocess_image(image_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fix_scale=1.0):
    """
    Read image, resize to (width, height), normalize and convert to int8 format.
    Ensure this size matches the quantized model's input size.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.resize(image, (width, height))
    image = image.astype(np.float32)
    image = normalize_image(image)
    # 2) Scale according to DPU requirements, round and convert to int8
    image = (image * fix_scale).astype(np.int8)
    return image

#############################
# 6. Main function
#############################
def main(argv):
    if len(argv) != 4:
        print("usage: python3 model_inference.py <thread_number> <xmodel_file> <image_dir>")
        sys.exit(1)

    threadnum = int(argv[1])
    xmodel_path = argv[2]
    image_dir = argv[3]  # Get image directory from command line arguments

    # Read xmodel file to generate graph
    g = xir.Graph.deserialize(xmodel_path)
    subgraphs = get_child_subgraph_dpu(g)
    if len(subgraphs) != 1:
        print("Error: Expected only one DPU subgraph")
        sys.exit(1)
    dpu_subgraph = subgraphs[0]

    # Create DPU Runner list, one Runner per thread
    runners = []
    for i in range(threadnum):
        r = vart.Runner.create_runner(dpu_subgraph, "run")
        runners.append(r)

    # Get input tensor's fix_point for quantization preprocessing (for reference)
    in_fixpos = runners[0].get_input_tensors()[0].get_attr("fix_point")
    in_scale = 2 ** in_fixpos

    # Read test images and their ground-truth labels
    file_list = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    img_list = []
    gt_list = []
    for fname in file_list:
        path = os.path.join(image_dir, fname)
        label = extract_label_from_filename(fname)
        if label is None:
            print(f"Cannot extract label from filename {fname}, skipping")
            continue
        # Note: Using 224×224 for preprocessing, must match xmodel input
        # 1) First create float32 [0,1] image
        img_f = preprocess_image(path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fix_scale=in_scale)
        img_list.append(img_f)
        gt_list.append(label)
    
    if len(img_list) == 0:
        print("No test images found in directory", image_dir)
        sys.exit(1)

    print(f"Read {len(img_list)} test images in total.")

    # Calculate number of images per thread
    images_per_thread = len(img_list) // threadnum
    remaining_images = len(img_list) % threadnum

    threads = []
    time_start = time.time()
    
    # Assign images to each thread
    start_idx = 0
    for i in range(threadnum):
        # Calculate number of images for current thread
        thread_images = images_per_thread + (1 if i < remaining_images else 0)
        # Get image subset for current thread
        thread_img_list = img_list[start_idx:start_idx + thread_images]
        thread_gt_list = gt_list[start_idx:start_idx + thread_images]
        # Update start index
        start_idx += thread_images
        
        # Create and start thread
        t = threading.Thread(target=runModel, args=(runners[i], thread_img_list, thread_gt_list, thread_images))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    time_end = time.time()
    
    total_frames = global_total  # Total test samples
    accuracy = (global_correct / total_frames) * 100 if total_frames > 0 else 0.0
    total_time = time_end - time_start
    fps = total_frames / total_time
    print("FPS = %.2f, total frames = %d, time = %.6f s" % (fps, total_frames, total_time))
    print("Overall Accuracy = %.2f%% (%d/%d)" % (accuracy, global_correct, total_frames))

if __name__ == "__main__":
    main(sys.argv)
