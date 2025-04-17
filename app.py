# app.py
import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from skimage.filters import threshold_otsu
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import plotly.graph_objects as go

# --- Reproducibility ---
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Model Definition ---
class CheXNet(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(1024, num_classes)
    def forward(self, x):
        return self.densenet(x)

@st.cache_resource
def load_model():
    model = CheXNet().eval()
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Streamlit UI ---
st.title("Tuberculosis Detection from Chest X-ray")
st.write("Upload a chest X-ray image to detect TB regions and view 3D visualizations.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 1) Load and preprocess image
    img_pil_rgb  = Image.open(uploaded_file).convert('RGB')
    img_pil_gray = img_pil_rgb.convert('L').resize((224,224))
    img_gray     = np.array(img_pil_gray)

    st.image(img_pil_gray, caption='Loaded Chest X-ray (Grayscale)', use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    input_tensor = transform(img_pil_rgb).unsqueeze(0).to(device)

    # 2) Grad-CAM
    cam     = GradCAM(model=model, target_layers=[model.densenet.features[-1]])
    grads   = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(8)])[0]
    rgb_np  = input_tensor.detach().cpu().squeeze().permute(1,2,0).numpy()
    rgb_np  = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min())
    cam_img = show_cam_on_image(rgb_np, grads, use_rgb=True)

    st.image(cam_img, caption="Grad‑CAM Overlay (Infiltration)", use_column_width=True)

    # 3) Binary mask
    heatmap      = (grads * 255).astype(np.uint8)
    blur         = cv2.GaussianBlur(heatmap, (5,5), 0)
    thresh_val   = threshold_otsu(blur)
    binary_mask  = (blur > thresh_val).astype(np.uint8) * 255
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    st.image(cleaned_mask, caption="Cleaned TB Region Mask", use_column_width=True)

    # 4) 3D Plotly surface
    x = np.linspace(0,223,224)
    y = np.linspace(0,223,224)
    x, y = np.meshgrid(x, y)
    z = img_gray / 255.0 * 50
    tb_mask = cleaned_mask.astype(bool)

    fig1 = go.Figure([
        go.Surface(z=z, x=x, y=y, colorscale='viridis', opacity=0.9, showscale=False),
        go.Scatter3d(x=x[tb_mask], y=y[tb_mask], z=z[tb_mask]+2,
                     mode='markers', marker=dict(size=3, color='red'), name='TB Patch')
    ])
    fig1.update_layout(title="Interactive 3D Surface with TB Highlight",
                       scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Intensity'))
    st.plotly_chart(fig1)

    # 5) Overlay 2D
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.imshow(img_gray, cmap='bone')
    ax2.imshow(cleaned_mask, cmap='Reds', alpha=0.4)
    ax2.set_title("X‑ray with TB Mask Overlay")
    ax2.axis('off')
    st.pyplot(fig2)

    # 6) 3D Matplotlib Slice View
    from mpl_toolkits.mplot3d import Axes3D
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot_surface(x, y, z, cmap='bone', alpha=0.5)
    ax3.scatter(x[tb_mask], y[tb_mask], z[tb_mask] + 2, color='red', s=3, label='TB Patch')
    ax3.set_title("3D Slice of Chest X-ray with TB Mask")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Intensity")
    ax3.legend()
    st.pyplot(fig3)

    # 7) 3D TB Heatmap
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    tb_intensity = img_gray[tb_mask]
    z_vals = np.zeros_like(tb_intensity)
    ax4.scatter(x[tb_mask], y[tb_mask], z_vals, c=tb_intensity, cmap='hot', s=1, alpha=0.7)
    ax4.set_title("3D Heatmap of TB-Affected Regions")
    st.pyplot(fig4)
