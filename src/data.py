import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from scipy import signal
import random
import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AudioAugmentor:
    def __init__(self, config):
        """Initialize audio augmentation with configuration"""
        self.config = config['augmentation']
        self.sr = config['audio']['sample_rate']
        
    def add_noise(self, audio):
        """Add random noise to audio"""
        noise = np.random.randn(len(audio))
        return audio + self.config['noise_factor'] * noise
        
    def change_pitch(self, audio):
        """Apply pitch shifting to audio"""
        return librosa.effects.pitch_shift(
            audio, 
            sr=self.sr, 
            n_steps=self.config['pitch_shift_range']
        )
        
    def change_speed(self, audio):
        """Apply time stretching to audio"""
        return librosa.effects.time_stretch(
            audio,
            rate=self.config['speed_change_range']
        )
        
    def simulate_room(self, audio):
        """Simulate room acoustics"""
        reverb = np.exp(-np.linspace(0, 1, 8000))
        audio_reverb = signal.convolve(audio, reverb, mode='full')[:len(audio)]
        return (1 - self.config['room_reverb']) * audio + self.config['room_reverb'] * audio_reverb

    def augment(self, audio):
        """Apply random augmentations to audio"""
        if not self.config['enabled']:
            return audio
            
        augmented_audio = audio.copy()
        
        # Apply random augmentations
        for method in ['noise', 'pitch', 'speed', 'room']:
            if random.random() < 0.5:
                if method == 'noise':
                    augmented_audio = self.add_noise(augmented_audio)
                elif method == 'pitch':
                    augmented_audio = self.change_pitch(augmented_audio)
                elif method == 'speed':
                    augmented_audio = self.change_speed(augmented_audio)
                elif method == 'room':
                    augmented_audio = self.simulate_room(augmented_audio)
            
        return augmented_audio

class AudioPreprocessor:
    def __init__(self, config):
        """Initialize audio preprocessor with configuration"""
        self.config = config['preprocessing']
        self.sr = config['audio']['sample_rate']
        
    def remove_silence(self, audio):
        """Remove silent segments from audio"""
        if not self.config['remove_silence']:
            return audio
        return librosa.effects.trim(audio, top_db=self.config['vad_threshold'])[0]
        
    def normalize(self, audio):
        """Normalize audio amplitude"""
        if not self.config['normalize']:
            return audio
        return librosa.util.normalize(audio)
        
    def preemphasis(self, audio):
        """Apply pre-emphasis filter"""
        if not self.config['preemphasis']:
            return audio
        return signal.lfilter([1, -0.97], [1], audio)
        
    def process(self, audio):
        """Apply all preprocessing steps to audio"""
        audio = self.remove_silence(audio)
        audio = self.normalize(audio)
        audio = self.preemphasis(audio)
        return audio

class AudioFeatureExtractor:
    def __init__(self, config):
        self.sr = config['audio']['sample_rate']
        
    def extract_features(self, y):
        """Extract features from audio"""
        logger.info(f"Starting feature extraction, audio length: {len(y)}")
        features = {}
        
        # MFCC features
        logger.info("Extracting MFCC features...")
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # Spectral features
        logger.info("Extracting spectral centroid features...")
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Pitch features
        logger.info("Extracting pitch features...")
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500)
        features['f0_mean'] = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
        features['f0_std'] = np.nanstd(f0) if not np.all(np.isnan(f0)) else 0
        
        # Rhythm features
        logger.info("Extracting rhythm features...")
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        features['tempo'] = tempo
        
        # Convert to vector
        logger.info("Converting features to vector...")
        feature_vector = []
        for key in sorted(features.keys()):
            if isinstance(features[key], np.ndarray):
                feature_vector.extend(features[key])
            else:
                feature_vector.append(features[key])
                
        logger.info("Feature extraction completed")        
        return np.array(feature_vector)

class StoryDataset(Dataset):
    def __init__(self, audio_paths, labels, config, transform=True):
        self.audio_paths = audio_paths
        self.labels = labels
        self.transform = transform
        self.config = config
        
        # Using the new feature extractor
        self.feature_extractor = AudioFeatureExtractor(config)
        self.preprocessor = AudioPreprocessor(config)
        self.augmentor = AudioAugmentor(config)
        
        # Cache-related settings
        self.cache_dir = config['paths'].get('features_cache_dir', 'data/features_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if config['paths'].get('use_cache', True):
            self._preprocess_and_cache_all()
    
    def _get_cache_path(self, audio_path):
        """Get the cache file path for features"""
        # Use audio filename as cache filename
        filename = os.path.basename(audio_path)
        cache_name = f"{os.path.splitext(filename)[0]}_features.npy"
        return os.path.join(self.cache_dir, cache_name)
        
    def _preprocess_and_cache_all(self):
        """Preprocess and cache all audio files"""
        logger.info("Starting preprocessing and caching all audio files...")
        
        for audio_path in tqdm(self.audio_paths, desc="Processing audio files"):
            cache_path = self._get_cache_path(audio_path)
            
            if not os.path.exists(cache_path):
                try:
                    # Load audio
                    y, sr = librosa.load(audio_path, sr=self.config['audio']['sample_rate'])
                    
                    # Preprocess
                    y = self.preprocessor.process(y)
                    
                    # Extract features
                    features = self.feature_extractor.extract_features(y)
                    
                    # Save features
                    np.save(cache_path, features)
                    
                except Exception as e:
                    logger.error(f"Failed to process file {audio_path}: {str(e)}")
                    continue
        
        logger.info("All audio files processed")
        
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        cache_path = self._get_cache_path(audio_path)
        
        try:
            if self.config['paths'].get('use_cache', True) and os.path.exists(cache_path):
                # Load features from cache
                features = np.load(cache_path)
                
                # Apply feature augmentation if needed
                if self.transform:
                    features = self._augment_features(features)
                    
            else:
                # Process audio if no cache
                y, _ = librosa.load(audio_path, sr=self.config['audio']['sample_rate'])
                y = self.preprocessor.process(y)
                
                if self.transform:
                    y = self.augmentor.augment(y)
                    
                features = self.feature_extractor.extract_features(y)
            
            # Convert to tensors
            features_tensor = torch.tensor(features, dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            
            return features_tensor, label
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            raise RuntimeError(f"Failed to process {audio_path}: {str(e)}")
            
    def _augment_features(self, features):
        """Simple feature-level data augmentation"""
        if not self.config['augmentation']['enabled']:
            return features
            
        # Add random noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise
            
        # Random scaling
        if random.random() < 0.5:
            scale = random.uniform(0.95, 1.05)
            features = features * scale
            
        return features
        
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.audio_paths)