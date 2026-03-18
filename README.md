# Low-Resolution CCTV Face Enhancement (Classical CV)

## Student Details
- **Name:** Gowlikar Ajitesh
- **Roll No:** CS24M118
- **Course:** M.Tech CSE
- **Assignment:** Sentio POC – Face Enhancement  

---

## Problem Statement

The goal of this assignment is to enhance **low-resolution CCTV face images** using only **classical computer vision techniques**.

The pipeline must strictly follow the order:

1. Denoising  
2. CLAHE (Contrast Enhancement)  
3. Multi-step Upscaling  
4. Zone-based Sharpening  

The final output should:
- Improve **visual clarity**
- Enhance **facial details**
- Maintain **natural appearance**
- Resize output to **240 × 240**

---

## Approach

I designed a **multi-stage enhancement pipeline** using OpenCV and NumPy. Each stage improves a specific aspect of the image.

---

## Stage 1: Adaptive Denoising

- Used `cv2.fastNlMeansDenoisingColored`
- Noise level is estimated dynamically
- Denoising strength is adjusted based on noise

Removes noise while preserving edges.

---

## Stage 2: CLAHE in LAB Space

- Converted image to LAB color space
- Applied CLAHE only on the **L (luminance) channel**
- Clip limit adjusted based on image contrast

Improves contrast without affecting colors.

---

## Stage 3: Multi-Step Upscaling

- Used **Lanczos interpolation**
- Upscaled in **two steps (×2 → ×2)**
- Applied **luminance-only sharpening**
- Added **bilateral filtering** for smoothing

Enhances resolution while preserving details.

---

## Stage 4: Zone-Based Sharpening

- Used Haar Cascade for **face detection**
- Applied stronger sharpening to:
  - Eyes and nose region
  - Moderate sharpening to mouth
- Background is kept softer

Improves important facial features without over-sharpening.

---

## Final Normalization

- Face-centered cropping
- Square padding (if required)
- Final resize to **240 × 240**

Ensures uniform output format.

---

## Evaluation Metrics

- **Sharpness** (Laplacian Variance)
- **SSIM** (Structural Similarity Index)
- **Runtime per image**

---

## Results
![Output_Image](Output1.png)
- Processed faces           : 10
- Average sharpness (raw)   : 56.1881
- Average sharpness (enh.)  : 565.0951
- Average sharpness gain    : 508.9070
- Average SSIM              : 0.4141
- Average runtime / face    : 135.86 ms


### Observations

- Significant improvement in sharpness (~10× increase)
- Faces appear clearer and more detailed
- SSIM maintained at reasonable level (structure preserved)
- Efficient runtime (~135 ms per image)

---

## Project Structure

```
CS24M111_Kavyasri/
│
├── profiles/                  # Input images
├── enhanced_faces/            # Output images
├── solution.py                # Main implementation
├── enhancement_report.html    # Visual report (before/after)
├── evaluation_metrics.json    # Metrics data
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```
---

## How to Run

### Step 1: Create Virtual Environment
  
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```
### Step 3: Run the Program

```bash
python solution.py
```
### Step 4: View Report

```bash
xdg-open enhancement_report.html
```
---

## Output Files

- Enhanced Images: enhanced_faces/
- HTML Report: Visual comparison (before vs after)
- Metrics JSON: Detailed evaluation results
- 
---

## Note

- The assignment mentions a reference_identities/ folder for recognition accuracy.
- This folder was not provided in the dataset.
- Hence, evaluation is done using:
- Sharpness
- SSIM
- Runtime

---

## Key Improvements Added

- Adaptive denoising based on noise level
- Luminance-only sharpening (more natural results)
- Blockiness-aware smoothing
- Face-aware cropping and sharpening
- Multi-step upscaling for better quality
---
##  HTML View
![HTML_View_Image](HTML_fileview.png)

## Conclusion

- This project successfully enhances low-resolution CCTV face images using only classical computer vision techniques.

The pipeline:

- Improves clarity and detail
- Maintains natural appearance
- Produces consistent 240×240 outputs
- Runs efficiently

🙌 Final Note

- This implementation strictly follows assignment constraints and provides a balanced trade-off between sharpness and realism, making it suitable for practical face enhancement tasks.