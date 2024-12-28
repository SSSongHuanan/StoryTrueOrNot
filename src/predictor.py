import torch
import json
import logging
import os
import pandas as pd
from .data import AudioPreprocessor, AudioFeatureExtractor
from .models import ModelEnsemble
import librosa

logger = logging.getLogger(__name__)

class StoryPredictor:
    def __init__(self, model_path, config_path):
        """Initialize story predictor with model and configuration
        
        Args:
            model_path: Path to the trained model weights
            config_path: Path to the configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model with correct input size
        self.model = ModelEnsemble(
            input_size=self.config['model']['input_size'],
            config=self.config
        ).to(self.device)
        
        try:
            # Load model weights safely with weights_only option
            checkpoint = torch.load(
                model_path, 
                map_location=self.device,
                weights_only=True
            )
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Load from full checkpoint
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Load from state dict only
                self.model.load_state_dict(checkpoint)
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
        self.model.eval()
        
        # Initialize preprocessor and feature extractor
        self.preprocessor = AudioPreprocessor(self.config)
        self.feature_extractor = AudioFeatureExtractor(self.config)
        
    def predict(self, audio_path):
        """Predict story type for a single audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            tuple: (prediction, confidence, error_message)
                - prediction: Boolean, True for true story, False for deceptive
                - confidence: Float, prediction confidence score
                - error_message: String, error message if any, None otherwise
        """
        try:
            # Load and preprocess audio
            y, sr = librosa.load(audio_path, sr=self.config['audio']['sample_rate'])
            y = self.preprocessor.process(y)
            
            # Extract features
            features = self.feature_extractor.extract_features(y)
            
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                features_tensor = features_tensor.to(self.device)
                output = self.model(features_tensor)
                probability = output.item()
                prediction = probability > 0.5
            
            return prediction, probability, None
            
        except Exception as e:
            logger.error(f"Error predicting {audio_path}: {str(e)}")
            return None, None, str(e)

def batch_predict(audio_dir, attributes_file, model_path, config_path):
    """Perform batch prediction on multiple audio files
    
    Args:
        audio_dir: Directory containing audio files
        attributes_file: Path to the attributes CSV file
        model_path: Path to the model weights
        config_path: Path to the configuration file
        
    Returns:
        pandas.DataFrame: Prediction results including:
            - filename
            - language
            - true_story_type (if available)
            - predicted_story_type
            - confidence
            - error (if any)
    """
    predictor = StoryPredictor(model_path, config_path)
    
    # Read attributes file
    df = pd.read_csv(attributes_file)
    results = []
    
    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['filename'])
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            continue
            
        prediction, probability, error = predictor.predict(audio_path)
        
        result = {
            'filename': row['filename'],
            'language': row['Language'],
            'true_story_type': row['Story_type'],
            'predicted_story_type': 'True Story' if prediction else 'Deceptive Story',
            'confidence': probability,
            'error': error
        }
        results.append(result)
    
    # Save results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('prediction_results.csv', index=False)
    
    # Calculate statistics
    total = len(results_df)
    successful = len(results_df[results_df['error'].isna()])
    
    # Calculate accuracy if true labels are available
    correct = len(results_df[results_df['true_story_type'] == results_df['predicted_story_type']])
    accuracy = correct / total if total > 0 else 0
    
    logger.info(f"Prediction completed:")
    logger.info(f"Total samples: {total}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    return results_df