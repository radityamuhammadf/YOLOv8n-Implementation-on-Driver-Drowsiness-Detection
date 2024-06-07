# YOLOv8n Implementation on Driver Drowsiness Detection

An implementation of [Ultralytics - YOLOv8](https://github.com/ultralytics/ultralytics) repository for training on blink and yawn datasets to detect driver drowsiness.

## Table of Contents
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)

## About
This inference repository uses the transfer learning method from EfficientDet-D0 with the following training configuration:
- YOLOv8 Version : n version (smallest)
- Image input size: 512
- Learning rate: 0,001 (1e-3)
- Batch size: 16
- Epochs: 25
- Dataloader workers: 2 

The training was stopped at 25 epochs due to an increasing total loss (it's unclear if this is relevant). The three latest epochs of the training progress resulted in the following code snippets:
```
 Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/25      1.54G     0.7596      0.408     0.9653          9        512: 100%|██████████| 639/639 [03:16<00:00,  3.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 91/91 [00:26<00:00,  3.45it/s]
                   all       2903       1031      0.919      0.911      0.965      0.738

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/25      1.54G     0.7409     0.3979      0.959          3        512: 100%|██████████| 639/639 [03:17<00:00,  3.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 91/91 [00:27<00:00,  3.30it/s]
                   all       2903       1031      0.927       0.91      0.971      0.738

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/25      1.55G     0.7396      0.392     0.9534          3        512: 100%|██████████| 639/639 [03:18<00:00,  3.21it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 91/91 [00:27<00:00,  3.36it/s]
                   all       2903       1031      0.938      0.892      0.967      0.736
```

The final validation from the model was tested with 2903 images (20% of the dataset) and resulted in the following metrics: 
```py
                 Class     Images  Instances      Box(P          R      mAP50    mAP50-95)
                   all       2903       1031      0.904      0.924      0.972      0.741
           closed-eyes       2903        598      0.903       0.93      0.969      0.684
                  yawn       2903        433      0.905      0.919      0.975      0.798
```

## Installation
1. Clone this GitHub repository:
```bash
git clone https://github.com/radityamuhammadf/YOLOv8n-Implementation-on-Driver-Drowsiness-Detection.git
```
2. Change the current working directory (cwd):
```bash
cd YOLOv8n-Implementation-on-Driver-Drowsiness-Detection
```
3. Create a virtual environment:
```sh
python -m venv venv
```
4. Activate virtual environment:
```sh
python -m venv venv
```
5. Install all the dependencies and PyTorch
Note: Ensure your PyTorch configuration matches your CUDA version (e.g., if you're using CUDA 12.5, consider using [PyTorch for CUDA 12.4](https://pytorch.org/) ) 
```sh
pip install -r requriements.txt
```

## Usage
Run the inference code
1.  From the video input (could be accessed but still in development)
```sh
py video_input.py
```
2.  Live Detection
```sh
py live_detection.py
```
**Additional Information**
Some modules may not be listed in the requirements.txt file (it is unclear why they are missing even after using the pip freeze command). If you encounter a `ModuleNotFoundError`, you can install the missing module using the pip command.


