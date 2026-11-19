# License Plate Detection and Blurring

## Overview
This repository provides an automated computer vision workflow to detect vehicle license plates and apply privacy-preserving blurring using OpenCV and Python. It employs a pre-trained Haar Cascade classifier to locate license plate regions of interest (ROI) within input images. Detected plate regions are subsequently processed using median filtering to obscure identifying numbers and characters while maintaining the integrity of the surrounding image.

## Prerequisites
Ensure Python 3.8 or higher is installed. The required third-party libraries can be installed via `pip`:

- `opencv-python`: Image processing, Haar cascade detection, and filtering functions.
- `numpy`: Numerical array manipulations for image representations.
- `matplotlib`: Visualization and image comparison display.
- `notebook` (or `jupyterlab`): Environment for running Jupyter notebooks.

### Model Weights
The required pre-trained Haar Cascade model (`haarcascade_licence_plate_rus_16stages.xml`) is included directly in the repository root. No additional external weights or asset downloads are necessary.

## How to Run

1. **Clone the repository and enter the directory**:
   ```bash
   git clone https://github.com/<username>/License-Plate-Blur.git
   cd License-Plate-Blur
   ```

2. **(Optional) Set up a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install opencv-python numpy matplotlib notebook
   ```

4. **Launch the notebook**:
   ```bash
   jupyter notebook License-Plate-Detection.ipynb
   ```

5. **Execute the cells**:
   Run the notebook cells sequentially to load `car_plate.jpg`, run detection with `detect_plate()`, and generate the blurred output with `detect_and_blur_plate()`.

To run the pipeline as a standalone Python snippet:
```python
import cv2

# Load image and classifier
img = cv2.imread('car_plate.jpg')
plate_cascade = cv2.CascadeClassifier('haarcascade_licence_plate_rus_16stages.xml')

# Detect plates and apply median blur to the ROI
plate_rects = plate_cascade.detectMultiScale(img)
for (x, y, w, h) in plate_rects:
    roi = img[y:y+h, x:x+w]
    roi = cv2.medianBlur(roi, 35)
    img[y:y+h, x:x+w] = roi

# Save or display output
cv2.imwrite('car_plate_blurred.jpg', img)
```

## Results
The code produces outputs for three distinct visualization stages:
- **Detection**: The classifier detects the license plate and draws a red rectangular bounding box (`(0, 0, 255)`, 2-pixel thickness) around the region.
- **Privacy Blurring**: A median blur filter with a kernel size of 35 (`cv2.medianBlur(roi, 35)`) is applied exclusively to the region of interest, effectively obscuring all text and registration numbers.
- **Comparative Output**: A two-panel Matplotlib plot displays a side-by-side comparison labeled "Before" (original image) and "After" (image with the blurred license plate).

## References
- OpenCV Cascade Classifier Documentation: https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html
- Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. In *Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*.
- OpenCV Haar Cascades Model Repository: https://github.com/opencv/opencv/tree/master/data/haarcascades
