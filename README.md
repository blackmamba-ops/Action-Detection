# Action-Detection Master

Action Detection using MediaPipe and LSTM

This project demonstrates real-time action detection using hand gestures and poses captured through a webcam. It utilizes the MediaPipe library for holistic gesture detection and TensorFlow/Keras for building and training Long Short-Term Memory (LSTM) neural networks.

Features:

- Real-time detection of gestures and poses using the webcam.
- Data collection and preprocessing for training the LSTM model.
- Training of the LSTM neural network to recognize predefined actions.
- Evaluation of the model's performance using a confusion matrix and accuracy score.
- Real-time demonstration of gesture recognition using the trained model.

Getting Started:

Prerequisites:
- Python 3.7+
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- NumPy (`pip install numpy`)
- TensorFlow (`pip install tensorflow`)

Installation:
1. Clone the repository:
   git clone https://github.com/blackmamba-ops/Action-Detection.git
   cd action-detection

2. Install the required dependencies:
   pip install -r requirements.txt

Usage:
1. Run the `Action Detection.py` script:
   python Action Detection.py

2. A window will open, showing the webcam feed with detected gestures and poses.
3. Follow the on-screen instructions for collecting gesture data and training the LSTM model.
4. After training, the model will be evaluated and saved.
5. The real-time gesture recognition demo will automatically start after training.

Project Structure:
- `Action_Detection_Refined.py`: The main script for data collection, model training, and real-time demo.
- `MP_Data/`: Directory for storing collected gesture data in `.npy` files.
- `Logs/`: Directory containing TensorBoard logs.
- `models/`: Directory to store trained model weights.
- `demo.gif`: A GIF demonstrating the real-time gesture recognition demo.

Contributing:
Contributions are welcome! Please feel free to open issues and submit pull requests.

License:
This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

---

Created by blackmamba-ops
