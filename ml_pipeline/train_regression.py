from __future__ import annotations

from xgboost import XGBRegressor


def train_regressor(X_train, y_train) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        colsample_bytree=0.8,
        random_state=0,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    return model
