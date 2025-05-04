import cv2
import webbrowser
from deepface import DeepFace  # Facial Emotion Recognition
import time

def detect_mood_and_play_music():
    """
    Detects the user's mood using the webcam and plays corresponding music.
    Allows image capture when 'c' key is pressed.
    """

    # Initialize the webcam (camera index 0 for the default camera)
    cam = cv2.VideoCapture(0)  # Use the default webcam

    if not cam.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'c' to capture a picture, 'q' to quit.")

    # Define mood-to-music mapping with online links
    mood_music = {
        "happy": "https://open.spotify.com/track/4bHsxqR3GMrXTxEPLuK5ue",  # Happy song
        "sad": "https://open.spotify.com/track/1qOl11Z3WuTmkc7ZpAc6UL",    # Sad song
        "neutral": "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC", # Neutral song
        "angry": "https://open.spotify.com/track/7GhIk7Il098yCjg4BQjzvb",   # Angry song
        "surprise": "https://open.spotify.com/track/3dhjNA0jGA8vHBQ1VdD6vV", # Surprise song
        "fear": "https://open.spotify.com/track/2nGFzvICaeEWjIrBrL2RAx"     # Fearful song
    }

    previous_mood = None  # To avoid playing the same music repeatedly

    while True:
        # Capture a frame from the webcam
        ret, frame = cam.read()

        if not ret:
            print("Failed to capture frame from the camera. Exiting...")
            break

        # Display the frame in a window
        cv2.imshow("Mood Detector - Press 'c' to capture, 'q' to quit", frame)

        # Capture image when 'c' is pressed
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # If 'c' is pressed, capture the image
            print("Capturing image for mood detection...")

            # Analyze the mood using DeepFace
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                # Get the dominant emotion from the analysis
                mood = analysis[0]['dominant_emotion']
                print(f"Detected Mood: {mood}")

                # Check if the mood has changed and play new music accordingly
                if mood != previous_mood:
                    # Get the corresponding music link
                    music_link = mood_music.get(mood, mood_music["neutral"])  # Default to neutral mood

                    # Open the music link in the default web browser
                    print(f"Opening music for mood: {mood}")
                    webbrowser.open(music_link)

                    # Update previous mood
                    previous_mood = mood

            except Exception as e:
                print(f"Error during mood detection: {e}")

        # Exit the loop if 'q' is pressed
        if key == ord('q'):
            print("Exiting the program...")
            break

    # Release the webcam and close OpenCV windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_mood_and_play_music()
