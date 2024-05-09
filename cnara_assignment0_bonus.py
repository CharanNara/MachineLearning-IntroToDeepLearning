import streamlit as st
from torchvision import datasets
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import seaborn as sns
from torchvision import transforms
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, roc_auc_score, confusion_matrix, auc
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from torchinfo import summary
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.image as mpimg
import random
import os
from PIL import Image
import time
# from torchmetrics import Accuracy
import pandas as pd

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def summary(self):
        return summary(best_early_model)
    

transform = transforms.Compose([
    transforms.Resize((28, 28)),  
    transforms.Grayscale(),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5], std=[0.5])
])
st.set_page_config(page_title = 'OCTMNIST CLASSIFICATION')
st.title('OCTMNIST class detector')


df = st.file_uploader(label='Upload OCTMNIST sample image')
if df:
    st.write('Predicted image successfully')
    print(df)
    
    # Load and preprocess the image
    # image = Image.open(df)
    image = Image.open(df)

    transform = transforms.Compose([ 
    transforms.Grayscale(),  
    transforms.Resize((28,28)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
])
    image_tensor = transform(image).unsqueeze(0)

    
    
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Load the model
    best_early_model = cnn()
    best_early_model.load_state_dict(torch.load('best_early_model.pth'))
    # best_early_model.to(device)

    with torch.no_grad():
        output = best_early_model(image_tensor)
        print(output)
        _, predicted_val = torch.max(output, 1)
        print(predicted_val)
        resultContainer = st.empty()
        resultContainer.write(f"Class predicted for the given OCTMNIST image: {predicted_val[0]}")