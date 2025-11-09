# COVID-HRCT-DualStage-Pipeline
This repository, COVID-HRCT-DualStage-Pipeline, provides a complete deep learning workflow for COVID-19 analysis using HRCT images. It includes segmentation and classification models, feature extraction file, and an inference script for end-to-end evaluation and result generation.
## Files
- **segmentation.py**: Implements and trains the Attention SegNet model for lesion segmentation. 
- **feature_extraction.py**: Before classifier training, the segmentation model (`Att_Segnet.keras`) is used to predict lesion regions on both infected and non-infected datasets. The resulting feature arrays are saved via `feature_extraction.py` as `segmented_features.npz` and then used for classification training. 
- **classification.py**: Lightweight CNN classifier trained using segmented lesion outputs.  
- **inference.py**: Demonstrates end-to-end inference for a sample image using trained segmentation and classification models.
