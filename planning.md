### PLANNING.md

#### **1. High-Level Vision**

**Project Name:** EcoVision: Semantic Change Sentinel

**Elevator Pitch:** EcoVision is an AI-powered engine that analyzes time-series satellite imagery to detect, classify, and quantify environmental changes like deforestation and urban sprawl. Going beyond simple pixel differences, it provides semantic understanding (e.g., "forest loss") and generates interactive heatmaps with estimated sustainability impacts, such as carbon sequestration loss. Built on open data and designed for accessibility, it empowers NGOs, researchers, and local governments to monitor and respond to climate-related threats with greater speed and precision.

**Core Goal:** To democratize environmental monitoring by creating a lightweight, interpretable, and edge-deployable AI tool that translates raw satellite data into actionable sustainability insights.

---

#### **2. System Architecture**

The system is designed as a modular pipeline that proceeds from data ingestion to impact reporting.

1.  **Data Ingestion & Preprocessing:**
    *   **Source:** Publicly available satellite imagery datasets, primarily the Onera Satellite Change Detection (OSCD) dataset featuring Sentinel-2 image pairs.
    *   **Module:** A data loading and preprocessing module built using `TorchGeo` and `Rasterio`.
    *   **Process:**
        *   Loads bi-temporal image pairs (before/after).
        *   Performs band normalization to ensure consistent data distribution.
        *   Applies data augmentation (e.g., flips, brightness jitter) to improve model robustness.
        *   Handles geospatial projections and raster alignment using GDAL.

2.  **Core AI Model: Siamese Twin Transformer Network**
    *   **Concept:** A dual-branch network where two identical transformer-based encoders process the "before" and "after" images independently. This allows the model to learn rich, contextual features for each image.
    *   **Architecture:**
        *   **Shared-Weight Encoders:** Two transformer encoders (inspired by Vision Transformers) process image patches to capture spatial and spectral context.
        *   **Fusion Module:** A cross-attention mechanism fuses the encoded features, allowing the model to explicitly compare the two images and generate a "difference map."
        *   **Decoder:** A lightweight decoder network upsamples the difference map to produce a binary change mask and semantic labels (e.g., "built-up," "vegetation loss").
    *   **Interpretability:** Grad-CAM is integrated to generate visual explanations, highlighting which pixel changes most influenced the model's decision.

3.  **Sustainability Impact Layer**
    *   **Purpose:** To translate the detected changes into meaningful environmental metrics.
    *   **Process:** The change mask output from the model is fed into a calculation module.
    *   **Calculation:** For deforestation, the module calculates the area of vegetation loss and applies IPCC-based formulas to estimate the corresponding carbon emissions (e.g., `CO₂ = k × ΔA × f`).
    *   **Data Source:** Formulas and emission factors are sourced from IPCC documentation, with citations provided for transparency.

4.  **Deployment & User Interface**
    *   **Platform:** A standalone web application built with `Streamlit`.
    *   **Functionality:**
        *   Allows users to upload their own satellite image pairs (e.g., from Google Earth Engine).
        *   Runs inference using a pre-trained model.
        *   Visualizes the results: before/after images, an animated difference overlay, a semantic heatmap, and an impact dashboard with key metrics (e.g., "hectares lost," "estimated tons CO2 emitted").
    *   **Deployment:** The application is designed to run entirely locally on a standard laptop, with no cloud services required.

---

#### **3. Constraints & Considerations**

*   **Software-Only:** The project relies entirely on publicly available software and datasets. No custom hardware, sensors, or fieldwork is required.
*   **Local-First Execution:** All components—data processing, model training, and the demo application—must be executable on a standard developer laptop (with or without a GPU).
*   **Resource Constraints:** The model architecture and data handling are optimized for limited compute resources, using techniques like downsampling image tiles and mixed-precision training.
*   **Data Imbalance:** Change events are rare in the dataset (<5% of pixels). This is addressed using focal loss during training to ensure the model prioritizes these critical but infrequent changes.
*   **Interpretability:** A core design principle is to avoid a "black-box" model. Visual explanations and transparent sustainability calculations are integrated to build user trust.

---

#### **4. Tech Stack & Tools**

*   **Programming Language:**
    *   Python 3.10

*   **Machine Learning & AI:**
    *   **Core Framework:** PyTorch 2.1
    *   **Transformer Modules:** Hugging Face `Transformers` library
    *   **Geospatial ML:** `TorchGeo` (for dataset loading and handling)
    *   **Computer Vision:** `Torchvision` (for data augmentation)
    *   **Advanced Segmentation:** `Segment Anything Model (SAM)` (for potential boundary refinement)

*   **Data Handling & Geospatial:**
    *   **Core Libraries:** NumPy, Pandas
    *   **Geospatial I/O:** `Rasterio`
    *   **Projections & Utilities:** GDAL

*   **Deployment & Visualization:**
    *   **Web App Framework:** Streamlit
    *   **Plotting:** Matplotlib, Seaborn

*   **Development & Version Control:**
    *   **Environment:** Jupyter Notebooks (for experimentation)
    *   **Code Hosting:** GitHub

*   **External Data/APIs:**
    *   **CO2 Calculations:** IPCC API wrappers or manually implemented formulas based on IPCC documentation.

***

