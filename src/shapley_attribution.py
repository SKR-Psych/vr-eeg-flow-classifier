import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

class ShapleyAttributionEngine:
    """
    Shapley Value Feature Attribution & Composite Biomarker Engine.
    Computes exact/tree/kernel SHAP attributions for EEG biomarkers,
    ranks features by predictive contribution, selects top biomarkers,
    and synthesizes the Shapley Composite Biomarker Index (SCBI).
    """

    def __init__(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame, feature_names: list = None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names if feature_names is not None else list(X_train.columns)
        self.shap_values = None
        self.explainer = None

    def compute_shap_values(self):
        """
        Fits appropriate SHAP explainer (TreeExplainer or general Explainer)
        and computes SHAP values on X_test.
        """
        try:
            # Try TreeExplainer first for tree models (RF, XGBoost, ExtraTrees)
            self.explainer = shap.TreeExplainer(self.model)
            shap_raw = self.explainer.shap_values(self.X_test)
        except Exception:
            # Fallback to shap.Explainer or KernelExplainer
            background = shap.sample(self.X_train, min(50, len(self.X_train)))
            try:
                self.explainer = shap.Explainer(self.model.predict_proba, background)
                shap_raw = self.explainer(self.X_test).values
            except Exception:
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
                shap_raw = self.explainer.shap_values(self.X_test)

        # Handle output format (multiclass vs binary vs list)
        if isinstance(shap_raw, list):
            # For binary classification list [shap_class0, shap_class1] -> take class 1 (flow prediction error)
            self.shap_values = np.array(shap_raw[1] if len(shap_raw) > 1 else shap_raw[0])
        elif isinstance(shap_raw, np.ndarray) and len(shap_raw.shape) == 3:
            self.shap_values = shap_raw[:, :, 1]
        else:
            self.shap_values = np.array(shap_raw)

        return self.shap_values

    def compute_global_importance(self) -> pd.DataFrame:
        """
        Computes global mean absolute SHAP value for each biomarker feature:
        |Phi_j| = (1/N) * sum_i |phi_{i,j}|
        """
        if self.shap_values is None:
            self.compute_shap_values()

        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        df_imp = pd.DataFrame({
            'biomarker': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)

        df_imp['importance_percentage'] = (df_imp['mean_abs_shap'] / df_imp['mean_abs_shap'].sum()) * 100.0
        return df_imp

    def select_top_biomarkers(self, top_k: int = 10) -> list:
        """
        Returns the top-K highest attributed biomarkers by mean absolute SHAP value.
        """
        df_imp = self.compute_global_importance()
        return df_imp['biomarker'].head(top_k).tolist()

    def compute_composite_biomarker_index(self, X: pd.DataFrame, y: np.ndarray = None, top_k: int = 10) -> np.ndarray:
        """
        Synthesizes the Shapley Composite Biomarker Index (SCBI):
        SCBI_i = sum_{j in TopK} w_j * Z(X_{i,j})
        where w_j = sign(corr(X_j, y)) * (|Phi_j| / sum |Phi|)
        """
        df_imp = self.compute_global_importance().head(top_k)
        top_cols = df_imp['biomarker'].tolist()
        shap_weights = df_imp['mean_abs_shap'].values
        norm_weights = shap_weights / np.sum(shap_weights)

        # Standardize features
        X_sub = X[top_cols].values
        means = np.mean(X_sub, axis=0, keepdims=True)
        stds = np.std(X_sub, axis=0, keepdims=True) + 1e-8
        Z = (X_sub - means) / stds

        # Determine directional sign from correlation with target (if y provided and matches len) or default
        if y is not None and len(y) == len(X):
            signs = np.array([np.corrcoef(X[c].values, y)[0, 1] for c in top_cols])
            signs = np.nan_to_num(signs, nan=1.0)
            signs = np.where(signs >= 0, 1.0, -1.0)
        else:
            signs = np.ones(top_k)

        final_weights = norm_weights * signs
        scbi_scores = np.dot(Z, final_weights)
        return scbi_scores

    def plot_shap_summary(self, save_path: str = None):
        """
        Generates and saves publication-quality SHAP beeswarm plot and feature importance bar chart.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 1. Bar Chart of Global Mean Absolute SHAP values
        df_imp = self.compute_global_importance().head(12)
        ax1.barh(df_imp['biomarker'][::-1], df_imp['mean_abs_shap'][::-1], color='#2b5c8f', alpha=0.85)
        ax1.set_xlabel(r'Mean Absolute SHAP Value |$\Phi$|', fontsize=12, fontweight='bold')
        ax1.set_title('Global Biomarker Shapley Attribution Ranking', fontsize=14, fontweight='bold', pad=12)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Annotate percentages
        for i, (val, pct) in enumerate(zip(df_imp['mean_abs_shap'][::-1], df_imp['importance_percentage'][::-1])):
            ax1.text(val + 0.001, i, f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold', color='#111111')

        # 2. Beeswarm Plot representation
        X_test_arr = self.X_test[self.feature_names].values
        top_indices = [self.feature_names.index(b) for b in df_imp['biomarker']]

        for i, idx in enumerate(top_indices[::-1]):
            shap_col = self.shap_values[:, idx]
            feat_val = X_test_arr[:, idx]
            norm_val = (feat_val - np.min(feat_val)) / (np.ptp(feat_val) + 1e-8)
            colors = plt.cm.coolwarm(norm_val)

            # Add jitter for scatter plot
            jitter = np.random.normal(0, 0.08, size=len(shap_col))
            ax2.scatter(shap_col, np.full_like(shap_col, i) + jitter, c=colors, s=15, alpha=0.7)

        ax2.set_yticks(range(len(df_imp)))
        ax2.set_yticklabels(df_imp['biomarker'][::-1], fontsize=11)
        ax2.set_xlabel('SHAP Value (Impact on Prediction Error / Flow)', fontsize=12, fontweight='bold')
        ax2.set_title('Biomarker Impact Distribution (Red=High, Blue=Low)', fontsize=14, fontweight='bold', pad=12)
        ax2.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[+] Saved SHAP attribution plot to: {save_path}")
        plt.close()
