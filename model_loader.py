from huggingface_hub import hf_hub_download
import joblib
from config import get_hf_token

class ModelLoader:
    _instance = None

    def __init__(self):
        token = get_hf_token()
        repo_id = 'skcept/crop-yield-prediction'
        self.model = joblib.load(hf_hub_download(repo_id=repo_id, filename='knn.joblib', token=token))
        self.le_region = joblib.load(hf_hub_download(repo_id=repo_id, filename='le_Region.joblib', token=token))
        self.le_soil = joblib.load(hf_hub_download(repo_id=repo_id, filename='le_Soil_Type.joblib', token=token))
        self.le_crop = joblib.load(hf_hub_download(repo_id=repo_id, filename='le_Crop.joblib', token=token))
        self.le_weather = joblib.load(hf_hub_download(repo_id=repo_id, filename='le_Weather_Condition.joblib', token=token))
        self.scaler = joblib.load(hf_hub_download(repo_id=repo_id, filename='minmax_scaler.joblib', token=token))

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance 