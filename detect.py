import cv2
import mediapipe as mp
import numpy as np
import pygame

mp_hands = mp.solutions.hands

def stopMewing():
    print("Stop Mewing")
    video = cv2.VideoCapture('stopit.mp4')
    pygame.mixer.init()
    pygame.mixer.music.load('stopit.mp3')

    pygame.mixer.music.play()
    
    while True:
        ret_v, frame_v = video.read()
        if not ret_v:
            break

        # Display video frame
        cv2.imshow('Video', frame_v)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def check_finger_on_lips(landmarks, lips_region,image_width, image_height):
    # Get the coordinates of the fingertip (index finger)
    index_fingertip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_fingertip_x, index_fingertip_y = index_fingertip.x * image_width, index_fingertip.y * image_height
    
    # Check if the fingertip is within the lips region
    lips_x1, lips_y1, lips_x2, lips_y2 = lips_region

    if lips_y1 < index_fingertip_y < lips_y2:
        return True
    else:
        return False

def calculate_bounding_box(hand_landmarks, image_width, image_height):
    x_values = [lm.x * image_width for lm in hand_landmarks.landmark]
    y_values = [lm.y * image_height for lm in hand_landmarks.landmark]
    min_x, min_y = min(x_values), min(y_values)
    max_x, max_y = max(x_values), max(y_values)
    return int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)

def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Perform face detection using Haar Cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Define the region of lips based on the detected face
                lips_region = (x, y + h // 2, x + w, y + h)
                
                # Draw the bounding box for the lips region (in red)
                #cv2.rectangle(frame, (lips_region[0], lips_region[1]), (lips_region[2], lips_region[3]), (0, 0, 255), 2)
            
            # Flip the frame horizontally for a mirror view
            frame = cv2.flip(frame, 1)

            # Get frame dimensions
            image_height, image_width, _ = frame.shape
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Hands
            results = hands.process(rgb_frame)
            
            # Extract the hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the bounding box of the hand
                    bbox = calculate_bounding_box(hand_landmarks, image_width, image_height)
                    # Draw the bounding box for the hand
                    #cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                    
                    # Check if finger is on lips
                    if check_finger_on_lips(hand_landmarks, lips_region, image_width, image_height):
                        stopMewing()
                        
            
            # Display the frame
            cv2.imshow('Frame', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
