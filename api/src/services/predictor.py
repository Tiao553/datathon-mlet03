from src.services.Feature_engineering import Feature_engineering
from src.services.Feature_engineering import engagement_score
from src.services.Feature_engineering import technical_score
from src.services.Feature_engineering import culture_score
from src.services.Feature_engineering import agregate_score


def predict_pipeline(data):
    data_cleaned = Feature_engineering(data)
    eng = engagement_score(data_cleaned)
    tech = technical_score(data_cleaned)
    cult = culture_score(data_cleaned)
    return agregate_score(eng, tech, cult)

    