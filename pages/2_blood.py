
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import io
import numpy as np


st.set_page_config(
    page_title="Распознавание клеток крови по фотографиям!",
)

st.write("# Здравствуйте! 👋")

st.sidebar.success("Загрузите свою фотографию.")

st.markdown(
    """
    Эта страница позволяет классифицировать изображение клеток. 
    В качестве обучающих данных использован датасет изображений клеток крови.
    Вы можете загрузить свое фото клеток крови и получить класс, к которому она относится.
"""
)

preprocessing_func = T.Compose(
    [
        T.RandomRotation(45),
        T.ColorJitter(),
        T.RandomHorizontalFlip(),
        T.Resize((224, 224)),
        T.ToTensor()
    ]
)

def preprocess(img):
    return preprocessing_func(img)


# Архитектура модели
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 4)

    def forward(self, x):
        return self.model(x)

@st.cache_resource()
def load_model():
    model = Classifier()
    model.load_state_dict(torch.load('savemodel.pt', map_location=torch.device('cpu')))
    model.eval()
    device = 'cpu'
    model.to(device)
    return model

model = load_model()

image = st.file_uploader('Загрузите фотографию')

def predict(img):
    img = preprocess(img)
    pred = model(img.unsqueeze(0).to(device))
    # делаем словарь, чтобы по индексу найти название класса
    labels = {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'}
    return st.write(labels[np.argmax(pred.detach().cpu().numpy())])

if image:
    image = Image.open(image)
    prediction = predict(image)
    st.image(image)
    st.write(prediction)

    