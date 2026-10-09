import cv2

# 0 is the default webcam; use 1, 2, etc. for external cameras
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame in a window
    cv2.imshow('Camera Window', frame)

    # Wait for 'q' key to close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close window
cap.release()
cv2.destroyAllWindows()