# **Action Detection Master**

(**Pretrained Action Detection.py**)

**Import Required Libraries:**

Import necessary libraries like cv2 for OpenCV, numpy for numerical operations, os for file and directory operations, mediapipe for the MediaPipe framework, and tensorflow for the machine learning part.

**Define Actions and Colors:**

Define the actions you want to recognize in the actions array.
Define colors for visualization.

**Function for Probability Visualization (prob_viz):**

This function visualizes the predicted probabilities as rectangles and text on the input frame.
It takes the prediction results (res), actions array, input frame, and colors as arguments.
It creates rectangles and text for each action based on the predicted probabilities.

**Function for MediaPipe Detection (mediapipe_detection):**

This function processes an image using the MediaPipe holistic model.
It converts the image to RGB format, processes it using the provided model, and then converts it back to BGR format for display.
It returns the processed image and the results of the detection.

**Function for Drawing Styled Landmarks (draw_styled_landmarks):**

This function draws the detected landmarks on an image with custom styles.
It takes the image and the results from the holistic model as arguments.
It uses mp_drawing to draw facial, pose, left hand, and right hand landmarks with different colors and styles.
Function for Extracting Keypoints (extract_keypoints):

This function extracts keypoints from the results of the holistic model and flattens them into an array.
It takes the results as input and extracts pose, face, left hand, and right hand keypoints.
The keypoints are stored in separate arrays (pose, face, lh, rh) and then concatenated into a single array using np.concatenate.

**Load Trained Model:**

Load the previously trained Keras model using tf.keras.models.load_model().

Main Loop for Real-Time Gesture Recognition:
Open the webcam (cv2.VideoCapture) for real-time video capture.
Inside the with block, create a Holistic instance from mp_holistic to perform holistic detection.
Initialize variables like sequence, sentence, predictions, and threshold.

**Loop through Frames:**

Use a loop to continuously read frames from the webcam.
Perform holistic detection using the mediapipe_detection function.
Draw styled landmarks on the image using the draw_styled_landmarks function.

**Keypoint Sequence Collection:**

Extract keypoints using the extract_keypoints function and append them to the sequence list.
Keep the sequence length fixed at 30 by appending new keypoints and discarding older ones.

**Make Predictions and Update Sentence:**

When the sequence reaches the desired length (30):
Use the loaded model to predict the action based on the keypoints sequence.
Update the predictions list with the predicted action index.
Check if the predicted action has been consistent for the last 10 frames and if its confidence is above the threshold.
Update the sentence with the recognized action.
Keep the sentence length within 5.

**Visualize Predictions and Sentence:**

Use the prob_viz function to visualize the predicted probabilities on the image.
Add a colored rectangle at the top of the image to display the recognized sentence.

**Display Image and Break Loop:**

Display the annotated image using cv2.imshow.
Exit the loop if the 'q' key is pressed.

**Release Resources and Close Windows:**

After exiting the loop, release the webcam capture and close all OpenCV windows using cap.release() and cv2.destroyAllWindows().
This code essentially captures frames from the webcam, processes them using the MediaPipe holistic model to detect keypoints of different body parts, and then uses a trained LSTM model to recognize gestures in real-time. The recognized gestures are displayed on the video feed along with probability visualization. The recognized gestures are updated based on a sequence of keypoints, ensuring more stable recognition.

***************************************************************************************
**You can use pre-trained weights from models trained on similar tasks for your gesture recognition model**

**actions = np.array(['hello', 'thanks', 'iloveyou'])** 

**Remember to add Actions too..**



