import coloredlogs
import logging

import pandas as pd
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential

logger = logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def validate_model(trained_model: Sequential, test_target: pd.DataFrame, target_features: pd.DataFrame):
    target_pred = trained_model.predict(target_features.values)
    r2 = r2_score(test_target, target_pred)
    logger.info(f"Prediction r2: {r2:.2}")
