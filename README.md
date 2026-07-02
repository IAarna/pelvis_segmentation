Pelvis CT Segmentation and Health Classification

Overview:

This project focuses on automatic **pelvis segmentation** from 3D CT scans and the subsequent **health classification** of the segmented pelvis. The segmentation model identifies the pelvic region from medical images, while the classification model determines whether the pelvis appears healthy or unhealthy based on extracted features.
The project is implemented entirely in **Python** using **PyTorch** and **MONAI**, with support for **nnU-Net v2** and **TotalSegmentator** as benchmark segmentation frameworks.



Features:

- 3D Pelvis CT image preprocessing
- Automatic pelvis segmentation
- Healthy vs. Unhealthy pelvis classification
- Medical image visualization
- Patch-based 3D training
- GPU acceleration
- Support for NIfTI (.nii/.nii.gz) datasets
- Easy-to-understand and modular code



Technologies Used:

- Python 3.10+
- PyTorch
- MONAI
- nnU-Net v2
- TotalSegmentator
- NumPy
- SciPy
- Scikit-Image
- Scikit-Learn
- Pandas
- Matplotlib
- SimpleITK
- NiBabel


Dataset:

The project expects CT scans in NIfTI format.
Example directory:


dataset/
    images/
        case001.nii.gz
        case002.nii.gz
    labels/
        case001.nii.gz
        case002.nii.gz




Workflow:

1. Load CT Scan
2. Preprocess Images
3. Segment Pelvis
4. Generate Binary Mask
5. Extract Pelvis Region
6. Extract Features
7. Classify Healthy / Unhealthy
8. Save Results
9. Visualize Prediction



Segmentation Models:

The project supports

- UNet
- UNETR
- SwinUNETR
- nnU-Net v2



Classification Models

- Logistic Regression
- Random Forest
- Support Vector Machine
- XGBoost (Optional)
- Simple Neural Network

Evaluation Metrics:

Segmentation

- Dice Score
- IoU (Intersection over Union)
- Hausdorff Distance
- Precision
- Recall

Classification

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC








