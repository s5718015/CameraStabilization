
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.models import load_model

# Config
MODEL_PATH = "model/jittery_camera_smoothing_model.h5"
X_PATH = "SyntheticData/X_jittery.npy"
Y_PATH = "SyntheticData/Y_smooth.npy"

def main():
    # Loads trained model without compiling
    model = load_model(MODEL_PATH, compile=False)

    # Load test data
    X = np.load(X_PATH)
    Y = np.load(Y_PATH)

    # Pick one sequence
    x = X[0:1]          
    y_true = Y[0]       
    y_pred = model.predict(x)[0]  # Predicted smooth path 

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Animated Camera Motion Smoothing")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # Dynamic limits based on data
    all_data = np.concatenate([x[0], y_true, y_pred])
    ax.set_xlim(np.min(all_data[:, 0]) - 0.1, np.max(all_data[:, 0]) + 0.1)
    ax.set_ylim(np.min(all_data[:, 1]) - 0.1, np.max(all_data[:, 1]) + 0.1)

    # Line objects
    jittery_line, = ax.plot([], [], 'r--', label='Jittery Input')
    ground_line, = ax.plot([], [], 'g-', label='Ground Truth Smooth')
    pred_line, = ax.plot([], [], 'b-', label='Predicted Smooth')
    ax.legend()

    # Animation function
    def update(frame):
        jittery_line.set_data(x[0][:frame+1, 0], x[0][:frame+1, 1])
        ground_line.set_data(y_true[:frame+1, 0], y_true[:frame+1, 1])
        pred_line.set_data(y_pred[:frame+1, 0], y_pred[:frame+1, 1])
        return jittery_line, ground_line, pred_line

    ani = animation.FuncAnimation(fig, update, frames=124, interval=60, blit=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
