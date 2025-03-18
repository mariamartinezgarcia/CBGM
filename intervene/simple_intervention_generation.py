import os
import sys

sys.path.append(".")

import argparse
import yaml

import torch
from torchvision.utils import save_image
from torch import nn
from models import cb_vaegan
import torchvision.transforms as transforms
import torch.nn.functional as F

import warnings

from utils.utils import sample_noise
import matplotlib.pyplot as plt

import wandb

warnings.filterwarnings("ignore", category=UserWarning)

def find_string_position(lst, target_string):
    return [index for index, value in enumerate(lst) if target_string in value]

def plot_images_side_by_side(image1, image2, title1="Active", title2="Inactive", wb=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(image2.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(title2)
    axes[1].axis('off')

    if wandb:
        wandb.log({ 'generated' : wandb.Image(fig)})
    
    plt.show()

def main(config):

    use_cuda = config["train_config"]["use_cuda"] and torch.cuda.is_available()

    wb = config['wandb']
    if wb:
        wandb.init(
            project = 'cbgm',
            entity = "mariamartinezga",
            config=config
        )
        
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if config["eval_config"]["checkpoint"]=='default':
        save_model_name = "./generation_checkpoints/cb_vaegan_" + config["dataset"]["name"]+'.pt'
    else:
        save_model_name = config["eval_config"]["checkpoint"]

    dataset = config["dataset"]["name"]

    if torch.cuda.is_available() and config["train_config"]["use_cuda"]:
        use_cuda = True
        device = torch.device("cuda")
    else:
        use_cuda = False
        device = torch.device("cpu")
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    torch.cuda.LongTensor if use_cuda else torch.LongTensor   

    model = cb_vaegan.cbGAN(config)
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:0")
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model.dec = nn.DataParallel(model.dec)
        model.enc = nn.DataParallel(model.enc)
        model.dis = nn.DataParallel(model.dis)
        model.dec.to(device)
        model.enc.to(device)
        model.dis.to(device)
    model.to(device)

    # Load Checkpoint
    model.dec.load_state_dict(torch.load(save_model_name))
    print("Model loaded from {}".format(save_model_name))

    ## Evaluation Mode ###
    model.eval()
    num_imgs = config['eval_config']['num_imgs'] # number of images we want to generate

    # 1. Generate noise vectors for generation
    noise = sample_noise(num_imgs, model.noise_dim, device=device)

    # 2. Generate concept prob vectors for generation
    # list of c concepts, each element is a tensor with concept_bins prob values
    concept_probs = []
    for i in len(config['model']['concepts']['concept_bins']):
        probs = torch.randn(num_imgs, config['model']['concepts']['concept_bins'][i])
        probs = F.softmax(probs, dim=1)
        concept_probs.append(probs.to(device))

    concept = config['eval_config']['concept_to_intervene']
    c = find_string_position(config['eval_config']['concept_to_intervene'], concept)
    
    # Note: this is a quick intervention on binary concepts!! Check how are they handling the categorical concept in the digits
    # 2.a First batch we fix one concept as active
    concept_probs_active = concept_probs.copy()
    concept_probs_active[c][:,0] = 0.001
    concept_probs_active[c][:,1] = 0.999
    # 2.b Second batch we fix that concept as inactive
    concept_probs_inactive = concept_probs.copy()
    concept_probs_inactive[c][:,1] = 0.001
    concept_probs_inactive[c][:,0] = 0.999

    # 3. Generate images with the concept active and inactive
    # 3.a Upload images to wandb by pairs (active/inactive)
    generated_active = model.dec.forward(noise, probs=concept_probs_active)
    generated_inactive = model.dec.forward(noise, probs=concept_probs_inactive)

    for i in num_imgs:
        title1 = 'Concept '+concept+' active'
        title2 = 'Concept '+concept+' inactive'
        plot_images_side_by_side(generated_active[i], generated_inactive[i], title1=title1, title2=title2, wb=wb)
    
    # 4. Save noise vectors and prob vectors for reproducibility
    os.makedirs("vectors_interventions/", exist_ok=True)
    torch.save(noise, 'noise_vectors_'+dataset+'_'+concept+'_.pt')
    torch.save(concept_probs_active, 'concept_probs_active_'+dataset+'_'+concept+'_.pt')
    torch.save(concept_probs_inactive, 'concept_probs_inactive_'+dataset+'_'+concept+'_.pt')
    print("Noise vectors and probability vectors saved.")

    '''
    # How to load the files
    noise = torch.load('noise_vectors.pt')
    concept_probs_active = torch.load('concept_probs_active.pt')
    concept_probs_inactive = torch.load('concept_probs_inactive.pt')
    '''