import cv2
import numpy as np;
import time

'''
# Initialize camera
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the frame with the detected faces
    cv2.imshow('Face Detection', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()


# Font to write text overlay
font = cv2.FONT_HERSHEY_SIMPLEX

# Create lists that holds the thresholds
hsvMin = (20,120,120)
hsvMax = (49,255,255)


# Adjust detection parameters
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 100;
 
# Filter by Area
params.filterByArea = True
params.minArea = 400
params.maxArea = 20000
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.5
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.5


# Get the camera for video capture
cap = cv2.VideoCapture(0)

# Duty cycle of the servo pwm signal
# We start with 1.5ms which is normally the center position
duty = 1500

while True:
    # Get a video frame
    _, frame = cap.read()
   
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
    # Apply HSV thresholds
    mask = cv2.inRange(hsv, hsvMin, hsvMax)
   
    # Erode and dilate
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
   
    # Detect blobs
    detector = cv2.SimpleBlobDetector_create(params)
   
    # Invert the mask
    reversemask = 255-mask
   
    # Run blob detection
    keypoints = detector.detect(reversemask)
   
    # Get the number of blobs found
    blobCount = len(keypoints)
   
    # Write the number of blobs found
    text = "Count=" + str(blobCount)
    cv2.putText(frame, text, (5,25), font, 1, (0, 255, 0), 2)
   

    if blobCount > 0:
        # Write X position of first blob
        blob_x = keypoints[0].pt[0]
        text2 = "X=" + "{:.2f}".format(blob_x )
        cv2.putText(frame, text2, (5,50), font, 1, (0, 255, 0), 2)
   
        # Write Y position of first blob
        blob_y = keypoints[0].pt[1]
        text3 = "Y=" + "{:.2f}".format(blob_y)
        cv2.putText(frame, text3, (5,75), font, 1, (0, 255, 0), 2)        
   
        # Write Size of first blob
        blob_size = keypoints[0].size
        text4 = "S=" + "{:.2f}".format(blob_size)
        cv2.putText(frame, text4, (5,100), font, 1, (0, 255, 0), 2)    
   
        # Draw circle to indicate the blob
        cv2.circle(frame, (int(blob_x),int(blob_y)), int(blob_size / 2), (0, 255, 0), 2)

        # Adjust the duty depending on where the blob is on the image
        # The assumption is that the image is 640 pixels wide. Then a
        # dead band of 60 pixels are created. When the blob is outside
        # of this range the duty cycle is adjusted in 10us steps  
        if blob_x > 320 + 60 and duty > 1000:
            duty = duty - 10
        if blob_x < 320 - 60 and duty < 2000:
            duty = duty + 10


    # Show image
    cv2.imshow("Blob detection", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()


# Font to write text overlay
font = cv2.FONT_HERSHEY_SIMPLEX

# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the cascade classifier has been loaded correctly
if face_cascade.empty():
    print("Error loading cascade classifier")
    exit()

# Get the camera for video capture
cap = cv2.VideoCapture(0)

while True:
    # Get a video frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Write the number of faces found
    faceCount = len(faces)
    text = "Count=" + str(faceCount)
    cv2.putText(frame, text, (5, 25), font, 1, (0, 255, 0), 2)

    # Display X, Y, and size for the first detected face
    if faceCount > 0:
        x, y, w, h = faces[0]
        face_info = f"X={x}, Y={y}, W={w}, H={h}"
        cv2.putText(frame, face_info, (5, 55), font, 1, (0, 255, 0), 2)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show image
    cv2.imshow("Face detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
'''


# Font for text overlay
font = cv2.FONT_HERSHEY_SIMPLEX

# Blob detection parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 7500
params.maxArea = 20000
params.filterByCircularity = False
params.minCircularity = 0.1
params.filterByConvexity = False
params.minConvexity = 0.5
params.filterByInertia = False
params.minInertiaRatio = 0.5

# Create a blob detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Get the camera for video capture
cap = cv2.VideoCapture(0)

# Read a frame to get the video dimensions
ret, frame = cap.read()
if ret:
    height, width = frame.shape[:2]
    segment_width = width // 3
    segment_height = height // 3

while ret:
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect blobs
    keypoints = detector.detect(gray)

    # Draw grid for visualization and number the segments
    for i in range(1, 3):
        cv2.line(frame, (segment_width * i, 0), (segment_width * i, height), (0, 0, 255), 2)
        cv2.line(frame, (0, segment_height * i), (width, segment_height * i), (0, 0, 255), 2)

    # Label the segments
    for i in range(3):
        for j in range(3):
            segment_number = i * 3 + j + 1
            text_x = segment_width * j + segment_width // 2
            text_y = segment_height * i + segment_height // 2
            cv2.putText(frame, str(segment_number), (text_x - 10, text_y + 10), font, 1, (255, 0, 0), 2)

    # Process each detected blob
    for keypoint in keypoints:
        blob_x = int(keypoint.pt[0])
        blob_y = int(keypoint.pt[1])
        blob_size = int(keypoint.size)

        # Determine the grid position
        grid_x = blob_x // segment_width
        grid_y = blob_y // segment_height

        # Calculate the segment number
        segment_number = grid_y * 3 + grid_x + 1

        # Draw circles around the blobs
        cv2.circle(frame, (blob_x, blob_y), blob_size // 2, (0, 255, 0), 2)

        # Display X, Y, Size, and Segment for the first blob in the corner
        info_text = f"X={blob_x}, Y={blob_y}, S={blob_size}, Segment={segment_number}"
        cv2.putText(frame, info_text, (5, 75), font, 0.5, (0, 255, 0), 2)
        break  # Only showing for the first blob for clarity

    # Show image
    cv2.imshow("Blob Detection", frame)

    # Get next frame
    ret, frame = cap.read()

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()


