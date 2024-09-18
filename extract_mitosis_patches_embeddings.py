import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ViTModel
from torchvision import transforms
import logging
from tqdm import tqdm
import argparse

# Custom Dataset class for loading images from a directory
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [os.path.join(root, name)
                            for root, dirs, files in os.walk(image_dir)
                            for name in files
                            if name.endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
# Function to extract features
def extract_features(dataloader, model):
    model.eval()
    all_features = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs = {"pixel_values": batch}
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
            all_features.append(features)

    all_features = torch.cat(all_features, dim=0)  # Concatenate all batches
    return all_features

parser = argparse.ArgumentParser(description='Select cancer type')
parser.add_argument('-c','--cancer', type=str, help='Experiment design to be processed')
args = parser.parse_args()
cancer = args.cancer

root_dir = "/home/u2070124/lsf_workspace/Data/Data/pancancer/mitosis_patches/"
save_root_dir = "/home/u2070124/lsf_workspace/Data/Data/pancancer/mitosis_patches_embeddings/"
os.makedirs(save_root_dir, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
# Create a file handler
handler = logging.FileHandler(f'{save_root_dir}/logs_phikon_{cancer}.log')
handler.setLevel(logging.INFO)
# Add the handlers to the logger
logger.addHandler(handler)

# set up the model
# Initialize the image processor and model
image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

# Get the expected size from the image processor
expected_size = image_processor.size["shortest_edge"] if "shortest_edge" in image_processor.size else 224

# Define the transformation for images
transform = transforms.Compose([
    transforms.Resize((expected_size, expected_size)),  # Resize images to the size expected by the model
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])


cases = os.listdir(f'{root_dir}/{cancer}/')
logger.info(f"Found total of {len(cases)} cases in Cancer: {cancer}")

for ci, case in enumerate(cases):
    # set the path
    target_root = f'{save_root_dir}/{cancer}/{case}/'
    source_root = f'{root_dir}/{cancer}/{case}/'

    if os.path.exists(target_root):
        logger.info(f"{ci}/{len(cases)} -- ALREADY EXISTS -- Cancer: {cancer} - Case: {case} ")
        continue

    # define dataset and dataloader
    dataset = ImageDataset(source_root, transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)

    if len(dataloader) > 0:
        logger.info(f"{ci}/{len(cases)} -- Working on Cancer: {cancer} - Case: {case} - #Files: {len(dataset)}")
        # Extract features from all images in the directory
        features_matrix = extract_features(dataloader, model)
    else:
        logger.info(f"{ci}/{len(cases)} -- EMPTY POINT LIST -- Cancer: {cancer} - Case: {case}")
        features_matrix = np.array([])

    
    os.makedirs(target_root, exist_ok=True)
    np.save(target_root+"phikon_embedding.npy", features_matrix)


