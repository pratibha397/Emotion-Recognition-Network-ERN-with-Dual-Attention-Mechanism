# Emotion-Recognition-Network-ERN-with-Dual-Attention-Mechanism
  Novel deep learning architecture for facial emotion recognition. Dual attention mechanism (spatial+channel) extracts comprehensive features. Emotion context modeling with hybrid loss. Achieved 74.45% test accuracy on AffectNet dataset. Exceptional results: Fear (86% F1), Surprise (78% F1), Contempt (77% F1).

# Emotion Recognition Network (ERN) with Dual Attention Mechanism

## Description
Novel deep learning architecture for facial emotion recognition using dual attention mechanisms and emotion context modeling to achieve state-of-the-art performance on the AffectNet dataset.

## Novelty
- **Dual Attention Mechanism**: Combines spatial attention (facial regions) and channel attention (feature channels) for comprehensive feature extraction
- **Emotion Context Modeling**: Learns prototypical emotion patterns and computes context similarities for refined predictions
- **Context-Aware Classification**: Hybrid loss function combining classification loss with KL divergence-based context alignment

## Dataset
- **AffectNet Dataset**: 25,262 total samples (17,101 training, 5,406 validation, 2,755 test)
- **8 Emotion Classes**: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt
- **Format**: YOLO format with image-label pairs

## Results
- **Test Accuracy**: 74.45%
- **Key Performance Metrics**:
  - Fear: 86% F1-score
  - Surprise: 78% F1-score
  - Contempt: 77% F1-score
- **Model Size**: 27.5 million parameters
- **Training**: 50 epochs with cosine annealing scheduling

## Installation and Usage
# Clone repository
git clone https://github.com/pratibha397/Emotion-Recognition-Network.git

# Install dependencies
pip install torch torchvision opencv-python pillow matplotlib pandas seaborn scikit-learn tqdm

# Run training
python emotion_recognition.py

## File Structure

├── emotion_recognition.py    # Main training script
├── best_emotion_model.pth    # Best model weights
├── final_emotion_model.pth   # Final model weights
├── evaluation_results.json   # Evaluation metrics
├── training_history.png      # Loss/accuracy curves
├── confusion_matrix.png      # Confusion matrix
├── context_similarities.png  # Context similarity visualization
├── feature_space.png         # t-SNE feature visualization
└── README.md                 # This file


## Key Features and Visualizations
1. **Training History**: Tracks loss and accuracy over 50 epochs
2. **Confusion Matrix**: Detailed performance across all emotion classes
3. **Context Similarities**: Visualization of emotion context relationships
4. **Feature Space**: t-SNE projection of learned feature representations
5. **Classification Report**: Precision, recall, and F1-scores for each emotion

## Why This Model is Novel
The ERN architecture introduces three key innovations:
1. **Integrated Attention Framework**: Simultaneous spatial and channel attention captures both local facial features and global feature importance
2. **Emotion Context Vectors**: Learnable prototypes that model relationships between different emotions
3. **Hybrid Loss Function**: Combines traditional classification with context alignment for more nuanced predictions

This integrated approach outperforms traditional single-attention models and provides interpretable insights into emotion recognition patterns.

## Contact
- **Author**: Pratibha Sharma
- **Email**: pratibha.sharmaa123@gmail.com
- **GitHub**: https://github.com/pratibha397

