from src.model import train_model
import numpy as np

def test_train_model():
    X = np.random.rand(10, 15)
    y = ["Happy"] * 5 + ["Calm"] * 5
    model = train_model(X, y)
    assert hasattr(model, "predict")
