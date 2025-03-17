import os
import sys

sys.path.append(".")
from datasets import color_mnist
import yaml
from utils.utils import get_dataset

import matplotlib.pyplot as plt
import torch

def show_images(images, n=8):
    plt.figure(figsize=(n*2, 2))
    for i in range(n):
        plt.subplot(1, n, i+1)
        img = images[i].permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
        plt.imshow(img)
        plt.axis("off")
    plt.show()

def main():

    with open('./config/cb_vaegan/confounded_color_mnist.yaml', "r") as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {'./config/cb_vaegan/confounded_color_mnist.yaml'}")

    dataloader = get_dataset(config)

    for batch in dataloader:
        images, labels = batch  # Assuming the dataloader returns (image, label)
        show_images(images, n=min(len(images), 8))  # Show up to 8 images
        break  # Remove this if you want to loop through the whole dataset




main()