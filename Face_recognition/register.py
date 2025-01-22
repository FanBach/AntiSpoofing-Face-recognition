import cv2
import os
from cvzone.FaceDetectionModule import FaceDetector

def register_user():
    # Prompt the user for a name
    name = input("Enter the name of the person to register: ").strip()

    if not name:
        print("Error: Name cannot be empty.")
        return

    # Create a folder for the user if it doesn't exist
    save_dir = f"data/{name}"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the FaceDetector
    detector = FaceDetector(minDetectionCon=0.7)

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print(f"Starting registration for {name}. Capturing 100 images...")

    count = 0  # To count the number of saved images

    while count < 100:
        # Read a frame from the webcam
        success, img = cap.read()
        if not success:
            print("Error: Unable to read from the camera.")
            break

        # Detect faces in the image
        img, bboxs = detector.findFaces(img, draw=True)

        # Save the image if a face is detected
        if bboxs:
            file_path = os.path.join(save_dir, f"{count}.jpg")
            cv2.imwrite(file_path, img)
            count += 1
            print(f"Image {count} saved at {file_path}")

        # Display the resulting frame with detected faces
        cv2.imshow("Registering - Press 'q' to quit", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration interrupted by user.")
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    if count == 100:
        print(f"Registration completed for {name}. {count} images saved.")
    else:
        print(f"Registration incomplete. Only {count} images saved.")

if __name__ == "__main__":
    register_user()
