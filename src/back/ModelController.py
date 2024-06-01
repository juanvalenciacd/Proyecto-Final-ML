import Definitions

import os.path as osp
import pandas as pd

from joblib import load # type: ignore
from keras.models import load_model # type: ignore
from src.model.TextPreprocessing import TextPreprocessing # type: ignore


class ModelController:

    def __init__(self):
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models", "text_classifier.h5")
        self.tfidf_path = osp.join(Definitions.ROOT_DIR, "resources/models", "trained_tfidf.joblib")
        self.pca_path = osp.join(Definitions.ROOT_DIR, "resources/models", "trained_pca.joblib")

        self.model = load_model(self.model_path)
        self.tfidf = load(self.tfidf_path)
        self.pca = load(self.pca_path)

        self.t_processing = TextPreprocessing()

    def predict(self, text):
        print("predict ->")

        x = self.t_processing.transform([text], self.tfidf, self.pca)
        """ Prediction probabilities """
        y_pred_prob = self.model.predict(x, verbose=0)
        df = pd.DataFrame(self.get_categories(), columns=["Category"])
        df['Probability'] = y_pred_prob.reshape(-1, 1)

        return df

    def get_categories(self):
        return ["Entertainment", "Business", "Technology", "Sports", "Education"]