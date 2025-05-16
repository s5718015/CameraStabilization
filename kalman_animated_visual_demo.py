
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.models import load_model

# Config
MODEL_PATH = "model/hybrid_kalman_lstm_model.h5"
X_PATH = "SyntheticData/X_kalman.npy"
Y_PATH = "SyntheticData/Y_smooth.npy"


def main():
    # Load model and data
    model = load_model(MODEL_PATH, compile=False)
    X = np.load(X_PATH)
    Y = np.load(Y_PATH)

    x = X[0:1]          
    y_true = Y[0]       
    y_pred = model.predict(x)[0]  

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Kalman + LSTM Camera Motion Smoothing")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # Setting axis limits based on all points so we can see everything
    all_data = np.concatenate([x[0], y_true, y_pred])
    ax.set_xlim(np.min(all_data[:, 0]) - 0.1, np.max(all_data[:, 0]) + 0.1)
    ax.set_ylim(np.min(all_data[:, 1]) - 0.1, np.max(all_data[:, 1]) + 0.1)

    # Creating empty lines for the animation
    kalman_input, = ax.plot([], [], 'r--', label='Kalman Input')
    ground_line, = ax.plot([], [], 'g-', label='Ground Truth Smooth')
    pred_line, = ax.plot([], [], 'b-', label='LSTM Prediction')
    ax.legend()

    # Updating the lines for animation
    def update(frame):
        kalman_input.set_data(x[0][:frame+1, 0], x[0][:frame+1, 1])
        ground_line.set_data(y_true[:frame+1, 0], y_true[:frame+1, 1])
        pred_line.set_data(y_pred[:frame+1, 0], y_pred[:frame+1, 1])
        return kalman_input, ground_line, pred_line

    ani = animation.FuncAnimation(fig, update, frames=124, interval=60, blit=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
