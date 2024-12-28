# StoryTrueOrNot

A deep learning system for classifying true and deceptive stories based on audio analysis. The system leverages ensemble learning with attention mechanisms and comprehensive audio feature extraction.

## Key Features

- Multi-modal feature extraction (MFCC, spectral, prosodic features)
- Ensemble learning with multiple attention-based models
- Advanced audio preprocessing pipeline
- Comprehensive data augmentation
- K-fold cross-validation training
- Wandb integration for experiment tracking
- GPU acceleration support
- Feature caching for improved performance

## System Requirements

- Python 3.10+
- CUDA compatible GPU (recommended)
- Required disk space: 
  - 10GB+ for audio files
  - 2GB+ for feature cache
  - 1GB+ for model checkpoints

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SSSongHuanan/StoryTrueOrNot.git
   cd story_classification
   ```

2. Create and activate conda environment:
   ```bash
   conda create -n story_clf python=3.10
   conda activate story_clf
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
story_classification/
├── data/                                  # Training data
│   ├── audio/                             # Audio files
│   ├── attributes.csv                     # Metadata
│   └── features_cache/                    # Cached features
├── inference_data/                        # Prediction data
│   ├── audio/                             # Test audio files
│   └── attributes.csv                     # Test metadata
├── models/                                # Model files
│   ├── best_model.pth                     # Best model weights
│   └── checkpoints/                       # Training checkpoints
├── src/                                   # Source code
│   ├── data.py                            # Data processing
│   ├── features.py                        # Feature extraction
│   ├── models.py                          # Model architecture
│   ├── predictor.py                       # Inference pipeline
│   ├── trainer.py                         # Training pipeline
│   └── logging_config.py                  # Logging configuration
├── logs/                                  # System logs
├── CBU5201_miniproject_submission.ipynb   # Jupyter notebook for training and prediction
├── config.json                            # Configuration
├── README.md                              # README file
└── requirements.txt                       # Dependencies
├── LICENSE                                # License
```

## Model Architecture

The system uses an ensemble of attention-based models:

- Multiple independent classifiers (configurable, default: 5)
- Multi-head attention mechanism (8 heads)
- Residual connections (4 layers)
- Adaptive feature fusion
- Focal loss for handling class imbalance

## Feature Extraction

Comprehensive audio feature extraction including:

1. Temporal Features
   - RMS energy
   - Zero crossing rate
   - Envelope features

2. Spectral Features
   - MFCC (40 coefficients)
   - Spectral centroid
   - Spectral bandwidth
   - Spectral rolloff

3. Prosodic Features
   - Fundamental frequency (F0)
   - Voice activity detection
   - Pitch statistics

## Data Augmentation

Rich augmentation options:
- Noise injection (configurable factor)
- Pitch shifting (±3 semitones)
- Time stretching (0.9-1.1x)
- Room reverberation
- SpecAugment (time/frequency masking)
- Background noise mixing

## Usage

### Training Mode

1. Prepare data:
   - Place audio files in `data/audio/`
   - Create `data/attributes.csv` with columns:
     - filename: Audio file name
     - Language: Audio language
     - Story_type: "True Story" or "Deceptive Story"

2. Configure training:
   - Adjust parameters in `config.json`
   - Key settings:
     ```json
     {
       "mode": "train",
       "model": {
         "num_models": 5,
         "hidden_size": 768
       },
       "training": {
         "batch_size": 16,
         "num_epochs": 50,
         "k_folds": 5
       }
     }
     ```

3. Run training:
   ```python
   python main.py
   ```

### Inference Mode

1. Prepare test data:
   - Place audio files in `inference_data/audio/`
   - Create `inference_data/attributes.csv`

2. Configure inference:
   ```json
   {
     "mode": "predict",
     "inference": {
       "ensemble_method": "weighted",
       "threshold": 0.5
     }
   }
   ```

3. Run prediction:
   ```python
   python main.py
   ```

## Experiment Tracking

The system uses Weights & Biases (wandb) for experiment tracking:

- Training metrics
- Learning curves
- Model performance
- System resources
- Hyperparameters

## Performance Optimization

1. Feature Caching:
   - Enable in config: `"use_cache": true`
   - Significantly reduces preprocessing time
   - Cached features stored in `data/features_cache/`

2. GPU Acceleration:
   - Automatic GPU detection
   - Mixed precision training
   - Batch size optimization

3. Memory Management:
   - Efficient feature loading
   - Gradient checkpointing
   - Memory-mapped file support

## Troubleshooting

1. Memory Issues:
   - Reduce batch_size in config.json
   - Enable feature caching
   - Reduce number of ensemble models

2. Training Issues:
   - Check audio file formats (WAV recommended)
   - Verify attributes.csv format
   - Monitor GPU memory usage

3. Performance Issues:
   - Enable feature caching
   - Adjust number of workers
   - Check GPU utilization

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## Performance Metrics

The system evaluates performance using multiple metrics:

1. Primary Metrics
   - F1 Score: Balanced measure of precision and recall
   - Accuracy: Overall classification accuracy
   - ROC-AUC: Area under ROC curve

2. Secondary Metrics
   - Precision: True positives / (True positives + False positives)
   - Recall: True positives / (True positives + False negatives)
   - Confusion Matrix: Detailed breakdown of predictions

## Model Training

1. Training Configuration
   - Batch size: 16 (configurable)
   - Learning rate: 1e-4 with cosine annealing
   - Optimizer: AdamW with weight decay
   - Loss function: Focal loss
   - Gradient clipping: 1.0

2. Training Process
   - K-fold cross-validation (k=5)
   - Early stopping with patience=10
   - Model checkpointing
   - Mixed precision training
   - Gradient accumulation support

## Inference Pipeline

1. Preprocessing
   - Audio format validation
   - Resampling to 22050Hz
   - Feature extraction and caching
   - Batch processing support

2. Ensemble Prediction
   - Weighted voting mechanism
   - Confidence thresholding
   - Batch inference support
   - GPU acceleration

## Error Handling

1. Audio Processing Errors
   - Invalid format handling
   - Corrupted file detection
   - Missing file handling
   - Length validation

2. Model Errors
   - Out of memory handling
   - Invalid input shape detection
   - NaN loss detection
   - GPU error recovery

3. System Errors
   - Disk space monitoring
   - Memory monitoring
   - GPU monitoring
   - Process recovery

## Best Practices

1. Data Preparation
   - Use high-quality audio recordings
   - Ensure consistent sampling rates
   - Remove background noise
   - Validate metadata format

2. Training
   - Start with default hyperparameters
   - Monitor training curves
   - Use feature caching
   - Enable early stopping

3. Inference
   - Batch process when possible
   - Use confidence thresholding
   - Monitor resource usage
   - Cache extracted features

## License

MIT License

Copyright (c) 2024 Huanan Song

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact

[song.hn2004@gmail.com]