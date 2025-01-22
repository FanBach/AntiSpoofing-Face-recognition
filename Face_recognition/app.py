import streamlit as st
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from cvzone.FaceDetectionModule import FaceDetector
from collections import deque

# Paths for saving models and label encoders
MODEL_PATH = 'face_recognition_model.h5'
ENCODER_PATH = 'label_encoder.pkl'

# Initialize face detector
detector = FaceDetector(minDetectionCon=0.7)

# Anti-spoofing tracking
face_positions = deque(maxlen=10)


# Function to capture images and save them
def capture_images(name):
    save_dir = f"data/{name}"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    st.write(f"Registering user '{name}'...")

    while count < 20:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to access the camera.")
            break

        frame, bboxs = detector.findFaces(frame, draw=True)
        if bboxs:
            file_path = os.path.join(save_dir, f"{count}.jpg")
            cv2.imwrite(file_path, frame)
            count += 1
            st.image(frame, channels="BGR", caption=f"Captured image {count}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if count == 20:
        st.success(f"Registration completed for '{name}'. 20 images saved.")
    else:
        st.error(f"Registration incomplete. Only {count} images saved.")


# Function to train the face recognition model
def train_model():
    st.write("Training the model...")

    # Directory containing face data
    data_dir = "data"

    # Data preprocessing and augmentation
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir, target_size=(160, 160), batch_size=32, class_mode='categorical', subset='training'
    )
    val_gen = train_datagen.flow_from_directory(
        data_dir, target_size=(160, 160), batch_size=32, class_mode='categorical', subset='validation'
    )

    # Build model
    base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(train_gen.num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_gen, validation_data=val_gen, epochs=10)

    # Save the model and label encoder
    model.save(MODEL_PATH)
    label_encoder = train_gen.class_indices
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)

    st.success("Model training completed. Ready for recognition.")


# Anti-spoofing function
def anti_spoofing(frame, bboxs):
    if not bboxs:
        return False

    for bbox in bboxs:
        x1, y1, x2, y2 = bbox['bbox']
        face_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        face_positions.append(face_center)

        if len(face_positions) < 2:
            return True

        movement = np.linalg.norm(np.array(face_positions[-1]) - np.array(face_positions[0]))
        if movement < 15:  # Movement threshold
            return False

    return True


# Recognize faces with anti-spoofing
def recognize_faces():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        st.error("Model or encoder not found. Train the model first.")
        return

    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

    cap = cv2.VideoCapture(0)
    st.write("Starting face recognition...")

    frame_placeholder = st.empty()  # Placeholder for displaying frames

    # Move the button outside of the loop to prevent duplication
    stop_recognition_button = st.button("Stop Recognition")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to access the camera.")
            break

        frame, bboxs = detector.findFaces(frame, draw=True)
        if bboxs:
            for bbox in bboxs:
                if not anti_spoofing(frame, bboxs):
                    cv2.putText(frame, "Spoof Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    continue

                x1, y1, x2, y2 = bbox['bbox']
                face = frame[y1:y2, x1:x2]

                # Ensure face region is not empty before resizing
                if face.size == 0:
                    continue  # Skip if no valid face region

                try:
                    # Resize face image to match model input size
                    face_resized = cv2.resize(face, (160, 160)).astype('float32') / 255.0
                    face_resized = np.expand_dims(face_resized, axis=0)

                    # Predict the identity of the face
                    predictions = model.predict(face_resized)
                    max_index = np.argmax(predictions[0])
                    predicted_label = list(label_encoder.keys())[list(label_encoder.values()).index(max_index)]

                    # Display prediction on the frame
                    cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                except Exception as e:
                    st.error(f"Error during face recognition: {e}")
                    continue

        # Update Streamlit UI with the current frame
        frame_placeholder.image(frame, channels="BGR")

        # Break the loop when the "Stop Recognition" button is pressed
        if stop_recognition_button:
            break

    cap.release()
    st.success("Face recognition stopped.")


# Streamlit UI
st.title("Face Recognition Workflow")

# Options for workflow
option = st.selectbox("Choose an action:", ["Register", "Train Model", "Recognize Faces"])

if option == "Register":
    name = st.text_input("Enter your name:")
    if name:
        capture_images(name)

elif option == "Train Model":
    if st.button("Start Training"):
        train_model()

elif option == "Recognize Faces":
    if st.button("Start Recognition"):
        recognize_faces()
