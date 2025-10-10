import coloredlogs
import logging

import pandas as pd
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential

logger = logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def validate_model(trained_model, test_target: pd.DataFrame, target_features: pd.DataFrame):
    test_target_numeric = test_target.drop(columns=["Player", "Position"]).apply(pd.to_numeric, errors='coerce').fillna(0)
    target_features_numeric = target_features.drop(columns=["Player", "Position"]).apply(pd.to_numeric, errors='coerce').fillna(0)
    
    if hasattr(trained_model, 'predict'):
        target_pred = trained_model.predict(target_features_numeric)
    else:
        target_pred = trained_model.predict(target_features_numeric.values)
    
    r2 = r2_score(test_target_numeric, target_pred)
    logger.info(f"Prediction r2: {r2:.2f}")
