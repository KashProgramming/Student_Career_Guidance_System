from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier


def train_classifier(X_train, y_train) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model
