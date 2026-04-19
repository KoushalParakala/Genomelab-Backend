"""
Pathogenicity Classifier Service
==================================
Lightweight ML classifier replacing pure rule-based pathogenicity predictions.
Uses a feature vector combining ESM embeddings, Grantham scores, structural 
metrics, and variant type to predict Pathogenic / Benign / VUS classifications.
"""

import numpy as np
import logging
import pickle
import os
from typing import Optional, Dict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "classifier_model.pkl")

# ─── Singleton Classifier ────────────────────────────────────────────

class _ClassifierHolder:
    """Lazy-loaded singleton for the pathogenicity classifier."""
    _instance: Optional["_ClassifierHolder"] = None
    
    def __init__(self):
        self.model: Optional[GradientBoostingClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self._loaded = False
    
    @classmethod
    def get(cls) -> "_ClassifierHolder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load(self):
        """Load or bootstrap the classifier."""
        if self._loaded:
            return
        
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    saved = pickle.load(f)
                self.model = saved['model']
                self.label_encoder = saved['label_encoder']
                self._loaded = True
                logger.info("Classifier loaded from disk")
                return
            except Exception as e:
                logger.warning(f"Failed to load saved classifier: {e}")
        
        # Bootstrap with biologically-informed synthetic data
        self._bootstrap_model()
        self._loaded = True
        logger.info("Classifier bootstrapped with synthetic training data")
    
    def _bootstrap_model(self):
        """Create initial classifier with synthetic training data based on biological heuristics."""
        np.random.seed(42)
        n_samples = 2000
        
        # Feature columns:
        # [embedding_distance, grantham_score, ddg_estimate, structural_rmsd,
        #  gc_content_delta, is_silent, is_missense, is_nonsense, is_frameshift]
        
        X = []
        y = []
        
        # --- Benign samples (Silent mutations, low impact missense) ---
        for _ in range(600):
            # Silent mutations
            X.append([
                np.random.uniform(0.0, 0.02),   # very low embedding distance
                0,                                 # grantham = 0 (same AA)
                np.random.uniform(-0.1, 0.1),     # neutral stability
                np.random.uniform(0.0, 0.2),      # minimal structural change
                np.random.uniform(-0.05, 0.05),   # gc content delta
                1, 0, 0, 0                         # silent
            ])
            y.append("Benign")
        
        for _ in range(200):
            # Low-impact missense
            X.append([
                np.random.uniform(0.01, 0.08),    # low embedding distance 
                np.random.uniform(0, 60),          # low grantham
                np.random.uniform(-0.5, 0.3),     # mild stability effect
                np.random.uniform(0.0, 0.5),      # low structural change
                np.random.uniform(-0.1, 0.1),
                0, 1, 0, 0                         # missense
            ])
            y.append("Benign")
        
        # --- Pathogenic samples ---
        for _ in range(300):
            # High-impact missense
            X.append([
                np.random.uniform(0.15, 0.5),     # high embedding distance
                np.random.uniform(100, 200),       # high grantham
                np.random.uniform(-4.0, -1.5),    # destabilizing
                np.random.uniform(1.5, 5.0),      # significant structural change
                np.random.uniform(-0.15, 0.15),
                0, 1, 0, 0
            ])
            y.append("Pathogenic")
        
        for _ in range(250):
            # Nonsense mutations
            X.append([
                np.random.uniform(0.2, 0.8),      # high embedding distance (truncation)
                np.random.uniform(100, 215),       # high grantham (if applicable)
                np.random.uniform(-5.0, -2.0),    # very destabilizing
                np.random.uniform(2.0, 8.0),      # severe structural change
                np.random.uniform(-0.2, 0.2),
                0, 0, 1, 0
            ])
            y.append("Pathogenic")
        
        for _ in range(250):
            # Frameshift mutations
            X.append([
                np.random.uniform(0.3, 0.9),
                np.random.uniform(50, 215),
                np.random.uniform(-5.0, -2.0),
                np.random.uniform(3.0, 10.0),
                np.random.uniform(-0.2, 0.2),
                0, 0, 0, 1
            ])
            y.append("Pathogenic")
        
        # --- VUS samples (intermediate) ---
        for _ in range(400):
            X.append([
                np.random.uniform(0.05, 0.2),     # moderate embedding distance
                np.random.uniform(40, 120),        # moderate grantham
                np.random.uniform(-2.0, -0.3),    # mild-moderate destabilization
                np.random.uniform(0.3, 2.0),      # moderate structural change
                np.random.uniform(-0.1, 0.1),
                0, 1, 0, 0
            ])
            y.append("VUS")
        
        X = np.array(X)
        y = np.array(y)
        
        # Add noise to make it realistic
        noise = np.random.normal(0, 0.02, X[:, :5].shape)
        X[:, :5] += noise
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X, y_encoded)
        
        # Save for future use
        try:
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'label_encoder': self.label_encoder
                }, f)
            logger.info(f"Classifier saved to {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to save classifier: {e}")


def ensure_classifier_loaded():
    """Pre-load classifier. Call during app startup."""
    _ClassifierHolder.get().load()


# ─── Prediction ──────────────────────────────────────────────────────

FEATURE_NAMES = [
    "embedding_distance",
    "grantham_score",
    "ddg_estimate",
    "structural_rmsd",
    "gc_content_delta",
    "is_silent",
    "is_missense",
    "is_nonsense",
    "is_frameshift"
]


def build_feature_vector(
    embedding_distance: float = 0.0,
    grantham_score: float = 0.0,
    ddg_estimate: float = 0.0,
    structural_rmsd: float = 0.0,
    gc_content_delta: float = 0.0,
    variant_annotation: str = "Missense"
) -> np.ndarray:
    """Construct the 9-dimensional feature vector for the classifier."""
    
    # One-hot encode variant type
    is_silent = 1.0 if variant_annotation == "Silent" else 0.0
    is_missense = 1.0 if variant_annotation == "Missense" else 0.0
    is_nonsense = 1.0 if variant_annotation == "Nonsense" else 0.0
    is_frameshift = 1.0 if "Frameshift" in variant_annotation or "Indel" in variant_annotation else 0.0
    
    return np.array([[
        embedding_distance,
        grantham_score,
        ddg_estimate,
        structural_rmsd,
        gc_content_delta,
        is_silent,
        is_missense,
        is_nonsense,
        is_frameshift
    ]])


def predict_pathogenicity(
    embedding_distance: float = 0.0,
    grantham_score: float = 0.0,
    ddg_estimate: float = 0.0,
    structural_rmsd: float = 0.0,
    gc_content_delta: float = 0.0,
    variant_annotation: str = "Missense"
) -> dict:
    """
    Predict pathogenicity using the ML classifier.
    
    Returns:
        dict with:
            - classification: str ("Pathogenic" | "Benign" | "VUS")
            - confidence: float (0-1)
            - probabilities: dict[str, float]
            - feature_importances: dict[str, float]
    """
    holder = _ClassifierHolder.get()
    if not holder._loaded:
        holder.load()
    
    features = build_feature_vector(
        embedding_distance=embedding_distance,
        grantham_score=grantham_score,
        ddg_estimate=ddg_estimate,
        structural_rmsd=structural_rmsd,
        gc_content_delta=gc_content_delta,
        variant_annotation=variant_annotation
    )
    
    try:
        probas = holder.model.predict_proba(features)[0]
        predicted_idx = np.argmax(probas)
        classification = holder.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(probas[predicted_idx])
        
        # Class probabilities
        classes = holder.label_encoder.classes_
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, probas)}
        
        # Feature importances
        importances = holder.model.feature_importances_
        feature_importances = {
            name: round(float(imp), 4) 
            for name, imp in zip(FEATURE_NAMES, importances)
        }
        
        return {
            "classification": classification,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "feature_importances": feature_importances,
        }
    
    except Exception as e:
        logger.error(f"Classifier prediction failed: {e}")
        # Fallback to rule-based
        return _fallback_prediction(variant_annotation, ddg_estimate)


def _fallback_prediction(variant_annotation: str, ddg: float) -> dict:
    """Rule-based fallback if ML classifier fails."""
    if variant_annotation == "Silent":
        return {"classification": "Benign", "confidence": 0.85, "probabilities": {"Benign": 0.85, "VUS": 0.10, "Pathogenic": 0.05}, "feature_importances": {}}
    if variant_annotation in ["Nonsense", "Frameshift / Indel"]:
        return {"classification": "Pathogenic", "confidence": 0.90, "probabilities": {"Pathogenic": 0.90, "VUS": 0.07, "Benign": 0.03}, "feature_importances": {}}
    if ddg < -2.0:
        return {"classification": "Pathogenic", "confidence": 0.70, "probabilities": {"Pathogenic": 0.70, "VUS": 0.20, "Benign": 0.10}, "feature_importances": {}}
    if ddg > -0.5:
        return {"classification": "Benign", "confidence": 0.65, "probabilities": {"Benign": 0.65, "VUS": 0.25, "Pathogenic": 0.10}, "feature_importances": {}}
    return {"classification": "VUS", "confidence": 0.50, "probabilities": {"VUS": 0.50, "Pathogenic": 0.30, "Benign": 0.20}, "feature_importances": {}}
