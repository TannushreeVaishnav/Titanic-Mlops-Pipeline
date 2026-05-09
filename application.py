import pickle
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.logger import get_logger
from src.feature_store import RedisFeatureStore
from sklearn.preprocessing import StandardScaler
from scipy import stats
from prometheus_client import start_http_server, Counter

logger = get_logger(__name__)

class KSDrift:
    """
    Kolmogorov-Smirnov drift detector.
    Mirrors the alibi-detect KSDrift.predict() response format.
    """
    def __init__(self, x_ref: np.ndarray, p_val: float = 0.05):
        self.x_ref = x_ref
        self.p_val = p_val

    def predict(self, x: np.ndarray) -> dict:
        p_values = []
        for col in range(self.x_ref.shape[1]):
            ref_col = self.x_ref[:, col].flatten()
            new_col = x[:, col].flatten()
            _, p = stats.ks_2samp(ref_col, new_col)
            p_values.append(p)

        p_values = np.array(p_values)
        is_drift = int(np.any(p_values < self.p_val))

        return {
            'data': {
                'is_drift': is_drift,
                'p_val': p_values,
            }
        }


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "artifacts\\model\\random_forest\\model.pkl"


app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

prediction_count = Counter('prediction_count', "Number of predictions made")
drift_count      = Counter('drift_count',      "Number of times data drift detected")
survived_count     = Counter('survived_count', "Number of predictions = Survived")
not_survived_count = Counter('not_survived_count', "Number of predictions = Did Not Survive")

print(f"[startup] Loading model from: {MODEL_PATH}")
print(f"[startup] Model file exists : {os.path.exists(MODEL_PATH)}")

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

print("[startup] Model loaded successfully.")

FEATURE_NAMES = [
    'Pclass','Sex','Age','Fare','Embarked','FamilySize','Isalone','HasCabin','Title','Pclass_Fare','Age_Fare'
]

feature_store = RedisFeatureStore()
scaler        = StandardScaler()


def fit_scaler_on_ref_data() -> np.ndarray:
    """Fit scaler on historical Redis data and return scaled reference array."""
    
    entity_ids   = feature_store.get_all_entity_ids()
    all_features = feature_store.get_batch_features(entity_ids)

    all_features_df = pd.DataFrame.from_dict(all_features, orient='index')


    print("[startup] Redis columns:", all_features_df.columns.tolist())

    all_features_df = all_features_df[FEATURE_NAMES]
    scaler.fit(all_features_df)
    return scaler.transform(all_features_df)


historical_data = fit_scaler_on_ref_data()
ksd = KSDrift(x_ref=historical_data, p_val=0.05)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        Age         = float(data["Age"])
        Fare        = float(data["Fare"])
        Pclass      = int(data["Pclass"])
        Sex         = int(data["Sex"])
        Embarked    = int(data["Embarked"])
        FamilySize  = int(data["FamilySize"])   
        Isalone     = int(data["Isalone"])
        HasCabin    = int(data["HasCabin"])
        Title       = int(data["Title"])
        Pclass_Fare = float(data["Pclass_Fare"])
        Age_Fare    = float(data["Age_Fare"])

        features = pd.DataFrame(
            [[Age, Fare, Pclass, Sex, Embarked,
              FamilySize, Isalone, HasCabin, Title,
              Pclass_Fare, Age_Fare]],
            columns=FEATURE_NAMES,
        )

        # Drift detection
        features_scaled = scaler.transform(features)
        drift           = ksd.predict(features_scaled)
        print("Drift Response:", drift)

        drift_response = drift.get('data', {})
        is_drift       = drift_response.get('is_drift', None)

        if is_drift == 1:
            print("Drift Detected.")
            logger.info("Drift Detected.")
            drift_count.inc()

        # Prediction
        prediction = model.predict(features)[0]
        prediction_count.inc()

        if prediction == 1:
            survived_count.inc()
            result='Survived'
        else:
            not_survived_count.inc()
            result='Did Not Survive'

        result = 'Survived' if prediction == 1 else 'Did Not Survive'
        return render_template(
            'index.html',
            prediction_text=f"The prediction is: {result}",
            survived=(prediction == 1),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest
    from flask import Response
    return Response(generate_latest(), content_type='text/plain')


if __name__ == "__main__":
    start_http_server(8000)
    app.run(debug=True, host='0.0.0.0', port=5000)