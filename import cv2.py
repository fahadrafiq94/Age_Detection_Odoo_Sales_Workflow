import cv2
import time

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

# Start timing
start_time = time.time()
frame_count = 0

print("Measuring FPS... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    frame_count += 1
    cv2.imshow('FPS Test', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

# Calculate average FPS
fps = frame_count / elapsed_time if elapsed_time > 0 else 0
print(f"\nAverage FPS: {fps:.2f}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
