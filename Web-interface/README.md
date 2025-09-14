# 🌊 Water Bodies Segmentation

This project provides an **AI-powered tool for water body detection and segmentation** from satellite or aerial images.  
It uses a deep learning model (ViT-based) integrated with a **Streamlit web interface** for an interactive experience.  

---

## 🚀 Features

- 📂 **Upload satellite/aerial images** (JPG, PNG)  
- 🖼️ **Preview input images** before processing  
- 🔍 **Segmentation outputs**:
  - Original Image  
  - Segmentation Mask  
  - Overlay Results  
- 🎚️ **Interactive controls**:
  - Threshold/sensitivity adjustment  
  - Visualization modes (Mask / Overlay / Boundary)  
- 📊 **Analysis & Statistics**:
  - Water vs. Land distribution pie chart  
  - Water area percentage  
  - Confidence score (from model)  
- ⬇️ **Download options**:
  - Segmentation Mask (PNG)  
  - Overlay Image (PNG)  
  - CSV Report (water/land pixel statistics)  
- 💡 **Fun fact box** to keep the interface engaging  
- ⚙️ **Advanced options**:
  - Batch image processing  
  - Map integration (optional with Folium/Leaflet)  

---

## 🖥️ Tech Stack

- **Python 3.9+**  
- [Streamlit](https://streamlit.io/) – Web app interface  
- [PyTorch](https://pytorch.org/) – Model training/inference  
- [OpenCV](https://opencv.org/) – Image processing  
- [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/) – Data handling  
- [Matplotlib](https://matplotlib.org/) – Visualization  

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/water-bodies-segmentation.git
cd water-bodies-segmentation
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

This will start a local web server at:

```
http://localhost:8501
```

Upload an image → adjust threshold/visualization → view results → download reports.  

---

## 📊 Example Output

- **Input Image**  
![Input](assets/sample_input.png)  

- **Segmentation Mask**  
![Mask](assets/sample_mask.png)  

- **Overlay Result**  
![Overlay](assets/sample_overlay.png)  

- **Analysis Pie Chart**  
![Pie](assets/sample_pie.png)  

---

## 📂 Project Structure

```
📁 water-bodies-segmentation
│── app.py                # Streamlit app
│── model/
│    └── vit_water.pth    # Trained ViT model (not included in repo)
│── notebooks/
│    └── VIT_Water_bodies.ipynb  # Model training & experimentation
│── assets/               # Sample images/screenshots
│── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

---

## 🔮 Future Improvements

- 🌍 Integration with GIS maps (Folium/Leaflet)  
- 🛰️ Support for satellite imagery metadata (GeoTIFF)  
- 📈 Dashboard view for multiple uploaded images  
- 🤖 Better water classification (shallow, deep, polluted, etc.)  

---

## 👩‍💻 Author

- **Your Name**  
📧 Email: your.email@example.com  
🔗 [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/)  

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use, modify, and share with attribution.  
