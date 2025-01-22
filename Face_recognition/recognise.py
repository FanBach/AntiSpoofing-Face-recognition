import cv2
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from collections import deque

# Load the pre-trained face recognition model and label encoder
model = load_model('face_recognition_model.h5')
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Anti-spoofing function with temporal analysis (tracks face movement over multiple frames)
face_positions = deque(maxlen=10)  # Store last 10 positions for movement tracking

# Function to estimate face distance (based on bounding box size)
def estimate_distance(bbox):
    # Use the bounding box height or width as a proxy for face distance.
    # Smaller bounding box size indicates the face is farther away.
    height = bbox[3] - bbox[1]
    distance = 600 / height  # Simple formula for distance estimation (tune for your camera)
    return distance

def anti_spoofing(frame, bboxs):
    # If no faces detected, return False for spoofing
    if not bboxs:
        return False
    
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox['bbox']
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue
        
        # Track the position of the face in the last few frames
        face_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        face_positions.append(face_center)

        # If face positions are too few, skip anti-spoofing
        if len(face_positions) < 2:
            return True  # Temporarily return True (not spoofed) as there's insufficient movement data

        # Calculate the movement between the first and last face positions
        movement = np.linalg.norm(np.array(face_positions[-1]) - np.array(face_positions[0]))

        # Estimate face distance from camera based on bounding box size
        distance = estimate_distance(bbox['bbox'])

        # Adjust the threshold based on face distance (closer faces need stricter checks)
        if distance < 10:  # If the face is very close (suggesting possible spoofing)
            movement_threshold = 10  # Lower threshold to detect subtle movements
        else:
            movement_threshold = 15  # Higher threshold for normal distance faces

        # If the movement is too small over the recent frames, it's likely spoofed
        if movement < movement_threshold:
            return False  # Spoof detected

    return True  # Face is live if it shows movement

# Recognize face and verify anti-spoofing
def recognize_face(frame, model, label_encoder):
    # Initialize detector
    detector = FaceDetector(minDetectionCon=0.7)
    img, bboxs = detector.findFaces(frame)

    if bboxs:
        for bbox in bboxs:
            x1, y1, x2, y2 = bbox['bbox']
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Anti-spoofing: Detect if the face is real or a spoof
            if not anti_spoofing(frame, bboxs):
                cv2.putText(frame, "Spoof Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue

            # Resize face to match the input size of the model (160x160 for MobileNetV2)
            face_resized = cv2.resize(face, (160, 160))  # Resize to the required input size
            face_resized = face_resized.astype('float32') / 255.0  # Normalize the pixel values
            face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension

            # Predict using the model
            predictions = model.predict(face_resized)
            max_index = np.argmax(predictions[0])
            predicted_label = label_encoder.inverse_transform([max_index])[0]
            confidence = predictions[0][max_index]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_label} ({confidence*100:.2f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return frame

# Main function for real-time face recognition with anti-spoofing
def main():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Starting face recognition with anti-spoofing...")

    while True:
        # Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            print("Error: Unable to read from the camera.")
            break

        # Recognize faces and apply anti-spoofing
        frame = recognize_face(frame, model, label_encoder)

        # Display the resulting frame with detected faces
        cv2.imshow("Face Recognition - Press 'q' to quit", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
