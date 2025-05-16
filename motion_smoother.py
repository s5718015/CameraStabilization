import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class MotionSmoother:
    def __init__(self, model_path, seq_length=124):
        self.seq_length = seq_length
        self.model = load_model(model_path, compile=False)

        # Trying to load a saved scaler used to normalise input values
        try:
            self.scaler = joblib.load("model/scaler.pkl")
        except Exception as e:
            print(f" Failed to load scaler: {e}")
            print("➡️ Using default MinMaxScaler (fit to [-10, 10, 10])")
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            dummy = np.array([[-10, -10, -10], [10, 10, 10]])
            self.scaler.fit(dummy)

        self.buffer = [] # This stores most recent imputs 
    # This function gets called with each new input point
    def predict(self, new_input):
        self.buffer.append(new_input)
        if len(self.buffer) < self.seq_length:
            return np.array([0.0, 0.0, 0.0])  
        elif len(self.buffer) > self.seq_length:
            self.buffer.pop(0)
        
        buffer_array = np.array(self.buffer)
        scaled = self.scaler.transform(buffer_array).reshape(1, self.seq_length, 3)
        prediction = self.model.predict(scaled, verbose=0)[0]

        # Return the last prediction in the sequence
        return prediction[-1]
