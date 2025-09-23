import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import io
import plotly.express as px

# --- Sidebar File Upload ---
st.markdown("## 🌊 Water Body Segmentation & Analysis")
uploaded_orig = st.file_uploader("Upload Original Image", type=["jpg", "jpeg", "png"], key="orig")
uploaded_mask = st.file_uploader("Upload Mask Image", type=["png"], key="mask")

orig_img, mask_img = None, None
if uploaded_orig: 
    orig_img = Image.open(uploaded_orig)
    st.image(orig_img, caption="Original", width=240)
if uploaded_mask: 
    mask_img = Image.open(uploaded_mask)
    st.image(mask_img, caption="Mask", width=240)

# --- Analysis Pie chart with Save ---
if mask_img:
    mask_np = np.array(mask_img.convert("L"))
    total = mask_np.size
    water_pixels = np.sum(mask_np > 0)
    land_pixels = total - water_pixels
    water_pct = round(100 * water_pixels / total, 1)
    land_pct = round(100 * land_pixels / total, 1)
    pie_df = pd.DataFrame({"Type": ["Water", "Land"], "Pixels": [water_pixels, land_pixels], "Pct": [water_pct, land_pct]})
    fig = px.pie(pie_df, names="Type", values="Pixels", color="Type", color_discrete_map={"Water": "#A3DAFB", "Land": "#8CF57B"}, hole=0)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Water:** {water_pct}%  |  **Land:** {land_pct}%")
    pie_df.to_csv("water_analysis.csv", index=False)
    st.success("Analysis saved to water_analysis.csv")

# # --- Overlay and Edge Buttons ---
# col1, col2 = st.columns([2,2])
# with col1:
#     if st.button("Show Predicted Overlay"):
#         if orig_img and mask_img:
#             # Overlay code: simple transparent blend
#             overlay = np.array(orig_img).copy()
#             msk = np.array(mask_img.convert("L"))
#             overlay[msk > 0] = [0, 170, 255]  # Example: blue color for water
#             overlay_img = Image.fromarray(overlay)
#             st.image(overlay_img, caption="Predicted Overlay", width=240)

#         # Canny/Sobel at left below overlay
#         if st.button("Canny on Predicted"):
#             canny = cv2.Canny(np.array(mask_img.convert("L")), 100, 200)
#             canny_rgb = np.dstack([canny]*3)
#             canny_overlay = cv2.addWeighted(np.array(orig_img).astype(np.uint8), 0.7, canny_rgb, 0.3, 0)
#             st.image(canny_overlay, caption="Canny on Predicted", width=240)
#         if st.button("Sobel on Predicted"):
#             sobel = cv2.Sobel(np.array(mask_img.convert("L")), cv2.CV_64F, 1, 0, ksize=5)
#             sobel = np.uint8(np.absolute(sobel))
#             sobel_rgb = np.dstack([sobel]*3)
#             sobel_overlay = cv2.addWeighted(np.array(orig_img).astype(np.uint8), 0.7, sobel_rgb, 0.3, 0)
#             st.image(sobel_overlay, caption="Sobel on Predicted", width=240)
# with col2:
#     if st.button("Show Canny + Sobel on Predicted Only"):
#         if mask_img:
#             canny = cv2.Canny(np.array(mask_img.convert("L")), 100, 200)
#             sobel = cv2.Sobel(np.array(mask_img.convert("L")), cv2.CV_64F, 1, 0, ksize=5)
#             sobel = np.uint8(np.absolute(sobel))
#             edge_mix = np.clip(canny.astype(np.float32) + sobel.astype(np.float32), 0, 255).astype(np.uint8)
#             st.image(edge_mix, caption="Canny+Sobel on Predicted", width=240)
