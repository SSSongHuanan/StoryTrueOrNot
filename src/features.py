import numpy as np
import librosa
import scipy
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    def __init__(self, config):
        """Initialize feature extractor with configuration"""
        self.config = config
        self.sr = config['audio']['sample_rate']
        
    def extract_features(self, y):
        """Extract comprehensive audio features
        
        Extracts multiple types of features:
        1. Temporal features
        2. Spectral features
        3. Pitch and tonal features
        4. Voice quality features
        5. Prosodic features
        """
        features = {}
        
        # 1. Basic temporal features
        features.update(self._extract_temporal_features(y))
        
        # 2. Spectral domain features
        features.update(self._extract_spectral_features(y))
        
        # 3. Pitch and tonal features
        features.update(self._extract_pitch_features(y))
        
        # 4. Voice quality features
        features.update(self._extract_voice_quality_features(y))
        
        # 5. Prosodic features
        features.update(self._extract_prosodic_features(y))
        
        # Convert features to vector
        feature_vector = np.array(list(features.values()))
        
        # Handle invalid values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_vector
    
    def _extract_temporal_features(self, y):
        """Extract temporal domain features
        
        Including:
        - RMS energy
        - Zero crossing rate
        - Envelope features
        """
        features = {}
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        features['rms_min'] = np.min(rms)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['zcr_skew'] = skew(zcr)
        
        # Envelope features
        envelope = np.abs(hilbert(y))
        features['env_mean'] = np.mean(envelope)
        features['env_std'] = np.std(envelope)
        
        return features
    
    def _extract_spectral_features(self, y):
        """Extract spectral domain features
        
        Including:
        - MFCC features
        - Spectral centroid
        - Spectral bandwidth
        - Spectral rolloff
        """
        features = {}
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_std'] = np.std(mfccs[i])
        
        # Spectral centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        features['spectral_centroid_mean'] = np.mean(cent)
        features['spectral_centroid_std'] = np.std(cent)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(bandwidth)
        features['spectral_bandwidth_std'] = np.std(bandwidth)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        features['spectral_rolloff_mean'] = np.mean(rolloff)
        features['spectral_rolloff_std'] = np.std(rolloff)
        
        return features
    
    def _extract_pitch_features(self, y):
        """Extract pitch-related features
        
        Including:
        - Fundamental frequency (F0) statistics
        - Voiced/unvoiced ratio
        - Voice probability features
        """
        features = {}
        
        # Extract fundamental frequency using pYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(y, 
            fmin=self.config['features']['prosodic']['pitch_min'],
            fmax=self.config['features']['prosodic']['pitch_max'],
            sr=self.sr)
        
        # F0 statistical features
        f0_cleaned = f0[~np.isnan(f0)]
        if len(f0_cleaned) > 0:
            features['f0_mean'] = np.mean(f0_cleaned)
            features['f0_std'] = np.std(f0_cleaned)
            features['f0_skew'] = skew(f0_cleaned)
            features['f0_kurtosis'] = kurtosis(f0_cleaned)
            features['f0_range'] = np.max(f0_cleaned) - np.min(f0_cleaned)
        else:
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['f0_skew'] = 0
            features['f0_kurtosis'] = 0
            features['f0_range'] = 0
        
        # Voice activity features
        features['voiced_ratio'] = np.mean(voiced_flag)
        features['voiced_prob_mean'] = np.mean(voiced_probs)
        
        return features
    
    def _extract_voice_quality_features(self, y):
        """Extract voice quality features
        
        Including:
        - Harmonic/percussive separation
        - Harmonic-to-noise ratio
        - Spectral flatness
        """
        features = {}
        
        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(y)
        features['harmonic_mean'] = np.mean(np.abs(harmonic))
        features['harmonic_std'] = np.std(np.abs(harmonic))
        features['percussive_mean'] = np.mean(np.abs(percussive))
        features['percussive_std'] = np.std(np.abs(percussive))
        
        # Harmonic-to-noise ratio
        S = np.abs(librosa.stft(y))
        features['hnr'] = np.mean(harmonic) / (np.mean(percussive) + 1e-8)
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['flatness_mean'] = np.mean(flatness)
        features['flatness_std'] = np.std(flatness)
        
        return features
    
    def _extract_prosodic_features(self, y):
        """Extract prosodic features
        
        Including:
        - Energy change rate
        - Pitch change rate
        - Tempo-related features
        """
        features = {}
        
        # Energy change rate
        rms = librosa.feature.rms(y=y)[0]
        rms_diff = np.diff(rms)
        features['energy_change_rate'] = np.mean(np.abs(rms_diff))
        
        # Pitch change rate
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=self.sr)
        f0_cleaned = f0[~np.isnan(f0)]
        if len(f0_cleaned) > 1:
            f0_diff = np.diff(f0_cleaned)
            features['pitch_change_rate'] = np.mean(np.abs(f0_diff))
        else:
            features['pitch_change_rate'] = 0
        
        # Tempo-related features
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr)
        features['tempo'] = tempo[0]
        
        return features
