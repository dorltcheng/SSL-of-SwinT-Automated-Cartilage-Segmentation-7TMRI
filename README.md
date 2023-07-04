# Self-Supervised Learning of Swin Transformer for Automated Cartilage Segmentation on 7 Tesla Knee MRI
Master's Thesis for MEng Biomedical Engineering

This study focuses on developing an automated cartilage segmentation framework for a limited amount of 7 Tesla (7T) ultra-high-resolution knee MRI. To address the small dataset problem, we utilised the technique of self-supervised learning (SSL) on the abundant unlabelled 3T knee MRI available from the [Osteoarthritis Initiative (OAI)](https://nda.nih.gov/oai/). The aim is to achieve the gold standard segmentations that facilitate future use in cartilage thickness quantification for early-stage KOA detection. Another key objective is to achieve data efficiency by maintaining model performance under a reduced amount of 7T MRI. The training process involves two stages, SSL pretraining (`SSLPretrain`) and segmentation fine-tuning (`Segmentation`). 


