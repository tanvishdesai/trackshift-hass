***

### **Phase 1: Machine Learning Model Development**

This phase focuses exclusively on data processing, model training, and evaluation. The single output is a trained model file. All work is designed to be completed within a single Kaggle notebook.

#### **PLANNING.md (Phase 1 - Machine Learning)**

**1. High-Level Vision**

*   **Objective:** To develop and train a high-performance semantic change detection model.
*   **Core Goal:** Produce a single, serialized model file (`ecovision_model.pth`) that can accurately identify and classify changes between two satellite images.
*   **Environment:** Kaggle Notebook (or Google Colab) with GPU acceleration.

**2. System Architecture**

*   **Data Source:** Onera Satellite Change Detection (OSCD) dataset, accessed via the `TorchGeo` library.
*   **Data Pipeline:**
    1.  **Load:** Download and instantiate the OSCD dataset.
    2.  **Preprocess:** Normalize image bands (RGB, SWIR1) to a standard distribution.
    3.  **Augment:** Apply random flips and brightness adjustments to create a more robust training set.
    4.  **Batch:** Use a `DataLoader` to feed 256x256 image patches in batches to the model.
*   **Model Architecture: Siamese Twin Transformer Network**
    1.  **Encoders:** Two identical, shared-weight transformer encoders process the "before" and "after" image patches.
    2.  **Fusion:** A cross-attention module compares the feature maps from the encoders to generate a difference representation.
    3.  **Decoder:** A lightweight convolutional decoder upsamples the difference map to produce a final change mask at the original resolution.
*   **Training Strategy:**
    *   **Loss Function:** A combination of Binary Cross-Entropy and Dice Loss to handle pixel-wise classification, with Focal Loss as an alternative to manage the high class imbalance (change vs. no-change).
    *   **Optimizer:** AdamW.
    *   **Epochs:** Approximately 50, with early stopping based on validation Intersection over Union (IoU).

**3. Constraints & Tech Stack**

*   **Code Structure:** All code (data loading, model definition, training loop, evaluation) will be contained within a **single Kaggle notebook** for maximum portability and reproducibility.
*   **Dependencies:** `torch`, `torchgeo`, `torchvision`, `rasterio`, `numpy`, `matplotlib`.
*   **Output Artifact:** A single file, `ecovision_model.pth`, containing the trained model's state dictionary. A secondary text file, `training_metrics.txt`, will log the final validation IoU and other relevant scores.

---

#### **TASK.md (Phase 1 - Machine Learning)**

**Milestone 1: Setup and Data Pipeline (Active)**

*   **[To Do]** Create a new Kaggle Notebook and enable the GPU accelerator.
*   **[To Do]** Install necessary libraries: `!pip install torchgeo rasterio`.
*   **[To Do]** Implement the data loading script using `TorchGeo` to download and prepare the OSCD dataset.
*   **[To Do]** Write and test the preprocessing and data augmentation functions.
*   **[To Do]** Set up `DataLoader` for both training and validation splits.

**Milestone 2: Model Architecture and Training Loop (Backlog)**

*   **[To Do]** Define the Siamese Transformer network as a `torch.nn.Module` class.
*   **[To Do]** Write the complete training and validation loop.
*   **[To Do]** Implement the chosen loss function and optimizer.
*   **[To Do]** Add metric tracking (IoU, accuracy) and progress visualization (plotting loss over epochs).

**Milestone 3: Execution and Finalization (Backlog)**

*   **[To Do]** Run the full training process until the model converges.
*   **[To Do]** Evaluate the final model on the held-out test set to get final performance metrics.
*   **[To Do]** Save the trained model's state dictionary to `ecovision_model.pth`.
*   **[To Do]** Save the final performance metrics to `training_metrics.txt`.
*   **[To Do]** Download the `ecovision_model.pth` file, which is the sole deliverable for this phase.

***
