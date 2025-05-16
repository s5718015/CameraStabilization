
### The video was submitted on Brightspace and uploaded to youtube 
### youtube link: https://youtu.be/PKKo2CNuOPs
# AI Camera Stabilisation for Virtual Production

This project uses deep learning to smooth out simulated camera motion, with a focus on real-time application in virtual production environments.

## Project Goal

To develop a lightweight camera motion smoothing system that can take jittery motion input and predict smooth, stabilized movement. The system uses:
- Synthetic motion data
- Classical Kalman filtering
- A neural network (LSTM) trained to enhance or replace traditional smoothing

## How It Works

The pipeline is as follows:
1. Generate synthetic jittery motion data
2. Pass input through a Kalman filter
3. Use an LSTM model to predict smoothed motion
4. Visualize and simulate output in real-time


## Folder Structure

```
MoSYs_Project_Submission/
├── model/                      # Trained models and scaler
│   ├── jittery_camera_smoothing_model.h5
│   ├── hybrid_kalman_lstm_model.h5
│   └── scaler.pkl            # scaler.pkl contains the MinMaxScaler fitted on the training data to ensure consistent normalization during model inference.
│
├── SyntheticData/             # Motion sequences
│   ├── X_jittery.npy
│   ├── X_kalman.npy
│   └── Y_smooth.npy
│          
├── Notebooks/             # Notebooks for training
│   ── train_lstm_raw_input.ipynb
│   ── train_lstm_kalman_input.ipynb      # Trains hybrid model using Kalman-filtered input
│
├── animated_visual_demo.py            # Animated prediction using raw model
├── kalman_animated_visual_demo.py     # Animated prediction using Kalman+LSTM model
├── realtime_ready_pipeline.py         # Prints real-time smoothing to terminal
│
├── Report/                    # Final report, video and visual assets
│   ├── CameraStabilizationReport.pdf
│   ├── VideoDemo.mp4
│   └── Graphs/
│
│ 
│
└── README.md                  
```

## Dependencies

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- joblib

Install dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run

### Train a Model
Use one of the Jupyter Notebooks:
- `train_lstm_raw_input.ipynb` for jittery input
- `train_lstm_kalman_input.ipynb` for Kalman-enhanced input


### View Animated Output
Run in terminal:
```bash
python animated_visual_demo.py
python kalman_animated_visual_demo.py
```

### Real-Time Simulation
```bash
python realtime_ready_pipeline.py
```

This simulates real-time frame-by-frame smoothing (print to terminal).

## Report & Video

The full methodology and system evaluation are in:
- `Report_Video_Graphs/CameraStabilizationReport.pdf`
- `Link on top of page`

---

## Author

**Glodis Ylja Hilmarsdottir Kjaernested**  
MSc Artificial Intelligence for Media  
Bournemouth University, 2025
