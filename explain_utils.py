"""Local weighted surrogate (LIME-method fallback) if the official lime package isn't installed.
This is explicitly an independent pedagogical approximation; prefer lime.lime_tabular when available.
"""
import numpy as np
from sklearn.linear_model import Ridge


def local_lime_surrogate(x, background, predict_proba, feature_names, n_samples=1800, n_features=10, seed=42):
    x = np.asarray(x, dtype=float).reshape(-1)
    bg = np.asarray(background, dtype=float)
    rng = np.random.default_rng(seed)
    ncols = x.size
    # Interpretable binary indicators: feature kept (1), otherwise replaced by empirical background value.
    keep = rng.binomial(1, 0.62, size=(n_samples, ncols)).astype(float)
    keep[0] = 1.0  # the point being explained
    bgrows = bg[rng.integers(0, len(bg), size=n_samples)]
    perturb = keep*x[None, :] + (1.0-keep)*bgrows
    p = np.asarray(predict_proba(perturb))[:, 1]
    # Similarity kernel over normalized Hamming distances, as in locally weighted surrogates.
    distance = np.sqrt(np.sum((1-keep)**2, axis=1) / ncols)
    weights = np.exp(-(distance**2)/(0.45**2))
    surrogate = Ridge(alpha=1.0)
    surrogate.fit(keep, p, sample_weight=weights)
    effects = surrogate.coef_
    top = np.argsort(np.abs(effects))[::-1][:n_features]
    fidelity = 1 - np.average((p - surrogate.predict(keep))**2, weights=weights) / max(np.average((p-np.average(p, weights=weights))**2, weights=weights), 1e-12)
    return [(str(feature_names[i]),float(effects[i])) for i in top],float(fidelity)


def lime_for_encoded(x, encoded_background, predict_proba, feature_names, n_features=10, seed=42):
    """Returns ([(feature, signed_effect_on_churn_probability)], method, fidelity_or_none)."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
        explainer = LimeTabularExplainer(
            np.asarray(encoded_background,dtype=float), mode='classification',
            feature_names=list(feature_names), class_names=['Permanece','Abandona'],
            discretize_continuous=False, random_state=seed)
        exp = explainer.explain_instance(
            np.asarray(x,dtype=float).reshape(-1), predict_proba,
            labels=(1,), num_features=n_features, num_samples=1800)
        pairs = [(str(name), float(effect)) for name,effect in exp.as_list(label=1)]
        return pairs,'LIME oficial',float(exp.score)
    except ImportError:
        pairs,fidelity=local_lime_surrogate(x,encoded_background,predict_proba,feature_names,
                                           n_features=n_features,seed=seed)
        return pairs,'LIME metodológico (alternativa local)',fidelity
