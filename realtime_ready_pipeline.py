
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time

# Config
MODEL_PATH = "model/hybrid_kalman_lstm_model.h5"
SEQUENCE_LENGTH = 124  # Number of frames the model expects
DIMENSIONS = 3 # Dimension like 3D

# Simple 1D kalman filter function
def simple_kalman_1d(z_series, R=0.01, Q=0.001):
    x_hat = z_series[0] # Initial estimate
    P = 1.0 # Initial errors
    filtered = [x_hat]
    for z in z_series[1:]:
        x_pred = x_hat # Predicts step
        P_pred = P + Q
        # Update step using Kalman gain
        K = P_pred / (P_pred + R)
        x_hat = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred

        filtered.append(x_hat)
    return np.array(filtered)

# Kalman filter 3D wrapper applies the Kalman filter to all 3 motion axes(x,y,z)
def apply_kalman_filter_3d(sequence):
    result = np.zeros_like(sequence)
    for axis in range(DIMENSIONS):
        result[:, axis] = simple_kalman_1d(sequence[:, axis])
    return result

# Simulated input source that mimics what would be real input
def generate_fake_jitter_frame(t):
    # Simulates a new jittery (x, y, z) frame based on time t
    jitter = np.random.normal(0, 0.3, size=(3,))
    motion = np.array([
        np.sin(0.1 * t) * 10 + jitter[0],
        np.cos(0.1 * t) * 10 + jitter[1],
        np.sin(0.05 * t) * 5 + jitter[2]
    ])
    return motion

# Main Loop
def main():
    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False) # Loads the LSTM model
    print("Model loaded. Starting real-time simulation...")

    buffer = deque(maxlen=SEQUENCE_LENGTH) # Stores last 124 frames
    t = 0

    while True:
        # Simulate new input point
        new_frame = generate_fake_jitter_frame(t)
        buffer.append(new_frame)
        t += 1
        # Starts prediction only after buffer is full
        if len(buffer) == SEQUENCE_LENGTH:
            # Convert buffer to numpy array
            sequence = np.array(buffer)

            # Apply Kalman filter
            kalman_input = apply_kalman_filter_3d(sequence)

            # Predict
            input_seq = np.expand_dims(kalman_input, axis=0)  # shape: (1, 124, 3)
            smoothed_seq = model.predict(input_seq, verbose=0)[0]

            # Get most recent prediction
            smoothed_point = smoothed_seq[-1]
            print(f"[{t}] Smoothed Output: {smoothed_point}")

        time.sleep(0.05)  # simulate 20 FPS (adjustable)

if __name__ == "__main__":
    main()
