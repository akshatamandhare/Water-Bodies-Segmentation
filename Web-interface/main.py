import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from io import BytesIO
import cv2
import pandas as pd

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Water Bodies Segmentation 🌊", layout="wide")

# ------------------- CUSTOM CSS -------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to bottom, #e0f7fa, #ffffff);
    }
    .title {
        text-align: center;
        font-size: 40px !important;
        font-weight: bold;
        color: #0277bd;
    }
    .subtitle {
        text-align: center;
        font-size: 18px !important;
        color: #004d40;
    }
    .funfact {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- HEADER -------------------
st.markdown("<div class='title'>🌊 Water Bodies Segmentation 💧</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload satellite images to detect and segment water bodies using our AI model</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------- UPLOAD SECTION -------------------
st.header("📂 Upload Section")
uploaded_file = st.file_uploader("Upload an Image (JPG, PNG) – Max 200MB", type=["jpg", "jpeg", "png"])

threshold = st.slider("Adjust Sensitivity/Threshold", min_value=0, max_value=255, value=127)
visualization_mode = st.selectbox("Choose Visualization Mode", ["Blue Mask", "Transparent Overlay", "Boundary Outline"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("🖼️ Preview")
    st.image(image, caption="Original Image", width=300)

    # ------------------- SEGMENTATION (PLACEHOLDER) -------------------
    # Replace this with your model's prediction
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Visualization options
    overlay = img_array.copy()
    if visualization_mode == "Blue Mask":
        overlay[mask == 255] = [0, 0, 255]   # blue for water
    elif visualization_mode == "Transparent Overlay":
        overlay = cv2.addWeighted(img_array, 0.7, np.dstack([mask, mask, mask]), 0.3, 0)
    elif visualization_mode == "Boundary Outline":
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # ------------------- DISPLAY RESULTS -------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(mask, caption="Segmentation Mask", use_container_width=True, clamp=True)
    with col3:
        st.image(overlay, caption="Overlay Result", use_container_width=True)

    # ------------------- METRICS -------------------
    water_pixels = np.sum(mask == 255)
    land_pixels = np.sum(mask == 0)
    total_pixels = water_pixels + land_pixels
    water_area_pct = (water_pixels / total_pixels) * 100

    st.header("📊 Analysis & Statistics")
    st.metric("Water Area Detected", f"{water_area_pct:.2f} %")
    st.metric("Confidence Score", "92 %")  # placeholder for model confidence

        # ------------------- PIE CHART 1: Water vs Land -------------------
    fig1, ax1 = plt.subplots(figsize=(3, 3), dpi=100)  # ~300px size
    ax1.pie(
        [water_pixels, land_pixels],
        labels=["Water", "Land"],
        autopct="%1.1f%%",
        colors=["#2196f3", "#a5d6a7"],
        textprops={"fontsize": 8}
    )
    ax1.set_title("Water vs Land", fontsize=10)

    # Center align in Streamlit
    col1, col2, col3 = st.columns([1, 2, 1])  # middle column is wider
    with col2:
        st.pyplot(fig1, clear_figure=True)


    # ------------------- DOWNLOAD OPTIONS -------------------
    st.header("⬇️ Download Results")
    # Save mask as image
    mask_pil = Image.fromarray(mask)
    buffer = BytesIO()
    mask_pil.save(buffer, format="PNG")
    st.download_button("Download Segmentation Mask", buffer.getvalue(), file_name="segmentation_mask.png", mime="image/png")

    # Save overlay
    overlay_pil = Image.fromarray(overlay)
    buffer2 = BytesIO()
    overlay_pil.save(buffer2, format="PNG")
    st.download_button("Download Overlay Image", buffer2.getvalue(), file_name="overlay.png", mime="image/png")

    # Save CSV report
    df = pd.DataFrame({"Category": ["Water", "Land"], "Pixels": [water_pixels, land_pixels], "Percentage": [water_area_pct, 100-water_area_pct]})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV Report", csv, "segmentation_report.csv", "text/csv")

    # ------------------- FUN FACT -------------------
    st.header("💡 Fun Fact")
    st.markdown(
        "<div class='funfact'>Did you know? 🌍 71% of Earth’s surface is covered by water, but only 2.5% is freshwater!</div>",
        unsafe_allow_html=True
    )

IMG_SIZE = 256
PATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_dim=512, dropout=0.1):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=256, depth=6, num_heads=8):
        super(Encoder, self).__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.img_size = img_size
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, heads=num_heads) for _ in range(depth)])

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, C, H', W']
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x, H, W

class Decoder(nn.Module):
    def __init__(self, embed_dim, output_channels=1):
        super(Decoder, self).__init__()
        self.mlp_head = nn.Linear(embed_dim, output_channels)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.mlp_head(x)  # [B, N, 1]
        x = x.transpose(1, 2).reshape(B, 1, H, W)  # [B, 1, H, W]
        x = torch.sigmoid(nn.functional.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear"))
        return x

class ViTSegmentation(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=256, depth=6):
        super(ViTSegmentation, self).__init__()
        self.encoder = Encoder(img_size, patch_size, in_chans, embed_dim, depth)
        self.decoder = Decoder(embed_dim)

    def forward(self, x):
        x, H, W = self.encoder(x)
        return self.decoder(x, H, W)

@st.cache_resource
def load_model():
    model = ViTSegmentation(img_size=IMG_SIZE, patch_size=PATCH_SIZE).to(DEVICE)
    model.load_state_dict(torch.load("web-interface\vit_waterbody.pth", map_location=DEVICE)
)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def predict(image: Image.Image):
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_t)
    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    return mask

def overlay_mask(image: Image.Image, mask: np.ndarray, color=(135, 206, 235)):
    img_np = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    overlay = img_np.copy()
    overlay[mask == 1] = np.array(color) / 255.0
    return (overlay * 255).astype(np.uint8)