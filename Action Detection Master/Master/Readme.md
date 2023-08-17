# **Action Detection Master**

**(Action Detection.py)**

The code provided is a comprehensive example of how to use the MediaPipe library and TensorFlow/Keras to perform real-time action recognition using hand gestures and poses captured through a webcam. I'll explain the key steps of the code step by step:

**Import Libraries:**

The necessary libraries are imported, including OpenCV (for computer vision tasks), NumPy (for numerical computations), TensorFlow/Keras (for building and training neural networks), MediaPipe (for holistic gesture detection), and other utilities.

**Initialize MediaPipe:**

mp_holistic and mp_drawing are initialized from the MediaPipe library. mp_holistic is used to detect holistic gestures like face, hands, and poses, while mp_drawing provides utilities to draw landmarks on images.

**Define MediaPipe Detection Functions:**

Two functions are defined: mediapipe_detection and draw_styled_landmarks. The first function takes an image and a MediaPipe model and returns the image with detected landmarks. The second function draws styled landmarks on an image.

**Open Webcam and Perform Detection:**

A loop captures video from the webcam using OpenCV.
Inside the loop, mediapipe_detection is used to detect gestures and poses in the captured frame. The detected results are then drawn on the frame using draw_styled_landmarks.
The processed frame is displayed using OpenCV's imshow.

**Extract Keypoints from Pose Results:**

The function extract_keypoints takes MediaPipe detection results and extracts keypoints for various body parts (pose, face, left hand, right hand). The keypoints are flattened and concatenated into a single array.

**Data Collection:**

This section collects gesture data by capturing frames from the webcam.
A loop iterates over the defined actions (gestures), sequences, and frames.
For each frame, the webcam captures an image, performs gesture detection using MediaPipe, and draws landmarks.
Detected keypoints are extracted using extract_keypoints, and these keypoints are saved as .npy files in appropriate directories.

**Preprocess Data and Create Labels:**

Gesture data (keypoints) is organized into sequences and labeled with corresponding action labels.
Data is organized into sequences with a fixed length (sequence_length).
Labels are encoded using one-hot encoding with to_categorical.

**Build and Train LSTM Neural Network:**

A Long Short-Term Memory (LSTM) neural network is defined using Keras' Sequential model.
The architecture consists of LSTM layers followed by dense layers.
The model is compiled with an optimizer, loss function, and evaluation metrics.
The model is trained on the preprocessed data using fit.

**Make Predictions and Save Model:**

The trained model is used to make predictions on test data.
The model's weights are saved to an HDF5 file using the model.save method.

**Load Model and Evaluate:**

The saved model's weights are loaded back into a new instance of the model using load_model.
The model's performance is evaluated using a confusion matrix and accuracy score.

**Test in Real Time:**

The last section demonstrates real-time gesture recognition using the trained model and webcam feed.
Detected gestures are displayed on the video feed using rectangles and text.
A threshold is applied to make predictions, and a history of predictions is maintained to improve accuracy.
Each section of the code contributes to the overall process of collecting gesture data, training a neural network, and using the trained model for real-time gesture recognition. The code is well-organized and commented, making it easy to follow each step of the process.





