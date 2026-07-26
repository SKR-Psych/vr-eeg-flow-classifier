import numpy as np
import pandas as pd
import mne
from mne.preprocessing import Xdawn
from mne.decoding import CSP

class SpatialFeatureExtractor:
    """
    Combines xDAWN spatial filtering (for ERP SNR maximization) and 
    Common Spatial Patterns (CSP, for frequency band power separation) 
    into a unified spatial feature extraction pipeline.
    """
    def __init__(self, n_xdawn_components: int = 4, n_csp_components: int = 4):
        self.n_xdawn_components = n_xdawn_components
        self.n_csp_components = n_csp_components
        
        self.xdawn = None
        self.csp_theta = None
        self.csp_alpha = None
        self.csp_beta = None
        self.is_fitted = False

    def fit(self, epochs: mne.Epochs, y: np.ndarray):
        """
        Fits xDAWN and CSP spatial filters ONLY on training data to prevent data leakage.
        """
        print(f"[*] Fitting Spatial Filters (xDAWN n={self.n_xdawn_components}, CSP n={self.n_csp_components})...")
        
        # 1. Fit xDAWN on time-domain ERPs with OAS covariance regularization
        self.xdawn = Xdawn(n_components=self.n_xdawn_components, reg='oas', correct_overlap=False)
        self.xdawn.fit(epochs)
        
        # 2. Filter epochs for CSP in frequency bands using IIR filters (suited for short epoch lengths)
        epochs_theta = epochs.copy().filter(l_freq=4.0, h_freq=8.0, method='iir', verbose=False)
        epochs_alpha = epochs.copy().filter(l_freq=8.0, h_freq=12.0, method='iir', verbose=False)
        epochs_beta = epochs.copy().filter(l_freq=12.0, h_freq=30.0, method='iir', verbose=False)
        
        # 3. Fit CSP on frequency bands with OAS regularization
        self.csp_theta = CSP(n_components=self.n_csp_components, reg='oas', log=True, norm_trace=False)
        self.csp_alpha = CSP(n_components=self.n_csp_components, reg='oas', log=True, norm_trace=False)
        self.csp_beta = CSP(n_components=self.n_csp_components, reg='oas', log=True, norm_trace=False)
        
        self.csp_theta.fit(epochs_theta.get_data(units='uV'), y)
        self.csp_alpha.fit(epochs_alpha.get_data(units='uV'), y)
        self.csp_beta.fit(epochs_beta.get_data(units='uV'), y)
        
        self.is_fitted = True
        print("[+] Spatial filters fitted successfully.")
        return self

    def transform(self, epochs: mne.Epochs) -> pd.DataFrame:
        """
        Transforms epochs into a multi-dimensional spatial feature vector.
        """
        if not self.is_fitted:
            raise RuntimeError("SpatialFeatureExtractor must be fitted before calling transform().")
            
        data_uv = epochs.get_data(units='uV')  # (n_epochs, n_channels, n_times)
        times = epochs.times
        n_epochs = len(epochs)
        
        feature_rows = []
        
        # 1. Transform xDAWN component signals -> (n_epochs, n_comp_total, n_times)
        xdawn_combined = self.xdawn.transform(epochs)
        
        pen_mask = (times >= 0.15) & (times <= 0.35)
        p300_mask = (times >= 0.30) & (times <= 0.50)
        
        # 2. Transform CSP features for frequency bands using IIR filtering
        epochs_theta = epochs.copy().filter(l_freq=4.0, h_freq=8.0, method='iir', verbose=False)
        epochs_alpha = epochs.copy().filter(l_freq=8.0, h_freq=12.0, method='iir', verbose=False)
        epochs_beta = epochs.copy().filter(l_freq=12.0, h_freq=30.0, method='iir', verbose=False)
        
        csp_theta_feats = self.csp_theta.transform(epochs_theta.get_data(units='uV'))
        csp_alpha_feats = self.csp_alpha.transform(epochs_alpha.get_data(units='uV'))
        csp_beta_feats = self.csp_beta.transform(epochs_beta.get_data(units='uV'))
        
        for i in range(n_epochs):
            feat = {}
            
            # Extract xDAWN component features
            comp_data = xdawn_combined[i]  # (n_comp_total, n_times)
            for c_idx in range(comp_data.shape[0]):
                feat[f'xdawn_comp_{c_idx}_pen_mean'] = float(np.mean(comp_data[c_idx, pen_mask]))
                feat[f'xdawn_comp_{c_idx}_p300_mean'] = float(np.mean(comp_data[c_idx, p300_mask]))
                feat[f'xdawn_comp_{c_idx}_pen_min'] = float(np.min(comp_data[c_idx, pen_mask]))
                feat[f'xdawn_comp_{c_idx}_p300_max'] = float(np.max(comp_data[c_idx, p300_mask]))
                
            # Extract CSP log-variance features
            for c_idx in range(self.n_csp_components):
                feat[f'csp_theta_comp_{c_idx}'] = float(csp_theta_feats[i, c_idx])
                feat[f'csp_alpha_comp_{c_idx}'] = float(csp_alpha_feats[i, c_idx])
                feat[f'csp_beta_comp_{c_idx}'] = float(csp_beta_feats[i, c_idx])
                
            feature_rows.append(feat)
            
        return pd.DataFrame(feature_rows)

    def fit_transform(self, epochs: mne.Epochs, y: np.ndarray) -> pd.DataFrame:
        """
        Fits spatial filters and transforms training epochs in one step.
        """
        return self.fit(epochs, y).transform(epochs)

if __name__ == "__main__":
    print("[*] Testing SpatialFeatureExtractor instantiation...")
    extractor = SpatialFeatureExtractor(n_xdawn_components=4, n_csp_components=4)
    print("[+] SpatialFeatureExtractor initialized successfully.")
