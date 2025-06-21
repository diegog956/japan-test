import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# --- Definición del modelo VAE ---
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(28*28 + 10, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 28*28)
    def encode(self, x, y):
        h1 = torch.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z, y):
        h3 = torch.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h3))
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# --- Cargar modelo ---
device = torch.device("cpu")
model = VAE().to(device)
model.load_state_dict(torch.load('vae_mnist.pth', map_location=device))
model.eval()

st.title("Generador de Dígitos Manuscritos (MNIST)")
digit = st.selectbox('Elige un dígito (0-9):', list(range(10)))
num_images = st.slider('Cantidad de imágenes a generar:', min_value=1, max_value=9, value=5)

if st.button("Generar"):
    images = []
    for _ in range(num_images):
        z = torch.randn(1, 20)
        y = torch.zeros(1, 10)
        y[0, digit] = 1
        with torch.no_grad():
            sample = model.decode(z, y).cpu().numpy().reshape(28,28)
        images.append(sample)

    st.write('Imágenes generadas:')
    cols = st.columns(num_images)
    for i, img in enumerate(images):
        pil_img = Image.fromarray((img*255).astype(np.uint8))
        cols[i].image(pil_img, use_container_width=True)

