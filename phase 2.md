

### **Phase 2: Web Development & Deployment**

This phase focuses on building the user-facing application using the model file created in Phase 1.

#### **PLANNING.md (Phase 2 - Web Development)**

**1. High-Level Vision**

*   **Objective:** To build an interactive web application that allows users to apply the EcoVision model to their own data.
*   **Core Goal:** Create an intuitive UI where a user can upload two images, run change detection, and see a visual report including a heatmap and sustainability impact metrics.

**2. System Architecture**

*   **Framework:** Streamlit (a pure Python framework).
*   **Frontend:**
    *   Title and project description.
    *   Two file uploader widgets for the "before" and "after" satellite images.
    *   A "Detect Change" button to trigger the analysis.
    *   Display areas for output: side-by-side input images, a change heatmap overlay, and a dashboard with metrics.
*   **Backend (all within Streamlit script):**
    1.  **Model Loading:** Load the `ecovision_model.pth` file from Phase 1 into memory when the app starts.
    2.  **Inference Pipeline:**
        *   Accept uploaded images from the user.
        *   Preprocess the images to match the format used during model training (e.g., normalization, resizing).
        *   Run the images through the loaded model to generate a change mask.
    3.  **Sustainability Layer:**
        *   Implement the `co2_impact` function that takes the generated change mask as input.
        *   This function calculates the area of change (e.g., vegetation loss) and applies an IPCC-based formula to estimate the CO2 impact in tons.
    4.  **Visualization:** Generate plots and metrics using Matplotlib or Streamlit's native components to display on the frontend.
*   **Database:** None required. The application is stateless.

**3. Constraints & Tech Stack**

*   **Input:** The application will use the `ecovision_model.pth` file from Phase 1. It does not perform any training.
*   **Dependencies:** `streamlit`, `torch`, `torchvision`, `numpy`, `pillow`, `matplotlib`.
*   **Deployment:** The final application will be deployed on Streamlit Community Cloud for public access.

---

#### **TASK.md (Phase 2 - Web Development)**

**Milestone 1: App Skeleton and Model Integration (Active)**

*   **[To Do]** Create the main `app.py` file.
*   **[To Do]** Write the script to load the `ecovision_model.pth` file.
*   **[To Do]** Build the basic Streamlit UI: title, file uploaders, and a button.

**Milestone 2: Inference and Visualization (Backlog)**

*   **[To Do]** Implement the function to handle user-uploaded images, including preprocessing them into tensors.
*   **[To Do]** Add the logic to run model inference when the user clicks the button.
*   **[To Do]** Implement the visualization logic to display the "before," "after," and generated heatmap/overlay images on the UI.

**Milestone 3: Sustainability Metrics and UI Polish (Backlog)**

*   **[To Do]** **Implement the `co2_impact` calculation function using IPCC-based formulas.**
*   **[To Do]** Integrate the output of this function into the UI, displaying metrics like "Hectares of Change Detected" and "Estimated CO2 Impact."
*   **[To Do]** Add clear instructions, explanations, and citations for the formulas used.
*   **[To Do]** Refine the layout and user experience.

**Milestone 4: Deployment (Backlog)**

*   **[To Do]** Create a `requirements.txt` file for the Streamlit app.
*   **[To Do]** Push the project (app code, requirements file, and the `.pth` model file via Git LFS) to a GitHub repository.
*   **[To Do]** Deploy the application from the GitHub repo to Streamlit Community Cloud.
*   **[To Do]** Test the live application and finalize the "Try It Out" link for submission.