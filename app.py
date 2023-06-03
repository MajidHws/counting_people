import cv2

# Load the video file
video_path = "feed.mp4"
cap = cv2.VideoCapture(video_path)

# Load pre-trained classifier for detecting people
cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
body_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize variables
total_people = 0

# Process each frame in the video
while cap.isOpened():
    # Read the current frame
    ret, frame = cap.read()

    # If the frame is not successfully read, exit the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people in the frame
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Update the total number of people
    total_people += len(bodies)

    # Display the frame with bounding boxes
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Print the total count of people
print("Total People:", total_people)
