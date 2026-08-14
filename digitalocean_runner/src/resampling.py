"""Training-fold-only imbalance methods, including conditional CTGAN."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import seed_everything

def resample(method: str, X: np.ndarray, y: np.ndarray, seed: int, feature_names: list[str], ctgan_epochs: int, log=print):
    seed_everything(seed)
    if method == "baseline": return X.copy(), y.copy()
    if method == "adasyn":
        from imblearn.over_sampling import ADASYN
        return ADASYN(sampling_strategy=1.0, n_neighbors=5, random_state=seed).fit_resample(X, y)
    if method == "borderline_smote":
        from imblearn.over_sampling import BorderlineSMOTE
        return BorderlineSMOTE(sampling_strategy=1.0, k_neighbors=5, m_neighbors=10, kind="borderline-1", random_state=seed).fit_resample(X, y)
    if method == "smote":
        from imblearn.over_sampling import SMOTE
        return SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=seed).fit_resample(X, y)
    if method == "smote_tomek":
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import TomekLinks
        smote = SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=seed)
        tomek = TomekLinks(sampling_strategy="all", n_jobs=1)
        return SMOTETomek(sampling_strategy=1.0, smote=smote, tomek=tomek, n_jobs=1, random_state=seed).fit_resample(X, y)
    if method != "ctgan": raise ValueError(f"unknown method: {method}")
    from sdv.metadata import SingleTableMetadata
    from sdv.sampling import Condition
    from sdv.single_table import CTGANSynthesizer
    labels = np.where(y == 1, "M", "B")
    training = pd.DataFrame(X, columns=feature_names); training.insert(0, "diagnosis", labels)
    needed = max(0, int((y == 0).sum() - (y == 1).sum()))
    if not needed: return X.copy(), y.copy()
    metadata = SingleTableMetadata(); metadata.detect_from_dataframe(training)
    metadata.update_column("diagnosis", sdtype="categorical")
    for column in feature_names: metadata.update_column(column, sdtype="numerical")
    synth = CTGANSynthesizer(metadata, epochs=ctgan_epochs, batch_size=50, pac=10, embedding_dim=128,
        generator_dim=(256, 256), discriminator_dim=(256, 256), generator_lr=0.0002, discriminator_lr=0.0002,
        generator_decay=1e-6, discriminator_decay=1e-6, discriminator_steps=1, log_frequency=True,
        verbose=False, enable_gpu=False)
    synth.fit(training)
    generated = synth.sample_from_conditions([Condition(num_rows=needed, column_values={"diagnosis": "M"})])
    if len(generated) != needed or not generated["diagnosis"].eq("M").all():
        raise RuntimeError("CTGAN conditional sample failed count or diagnosis validation")
    X_new = np.vstack([X, generated[feature_names].to_numpy(dtype=float)])
    return X_new, np.concatenate([y, np.ones(needed, dtype=int)])
