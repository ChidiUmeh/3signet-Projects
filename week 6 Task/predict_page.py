import streamlit as st
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, in_features=74, h1=50, h2=30, h3=20, h4=10, output_features=2):
        super().__init__()   # Instantiate our nn.Module
        self.fc1=nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,h3)
        self.fc4 = nn.Linear(h3,h4)
        self.out= nn.Linear(h4,output_features)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x= self.out(x)

        return x
    
def load_model():
    new_model = Model()
    new_model.load_state_dict(torch.load('my_pytorch_model.pt'))
    new_model.eval()
    return new_model

new_model =load_model()



def load_encoder():
    with open('saved_encode', 'rb') as file:
        encoding = pickle.load(file)
    return encoding
    
encoding = load_encoder()
label = encoding['label']
onehot = encoding['onehot']

def show_predict_page():
    st.title('Student Droout Risk Prediction')

    st.write("""### We need some information to predict Student Dropout Risk""")






