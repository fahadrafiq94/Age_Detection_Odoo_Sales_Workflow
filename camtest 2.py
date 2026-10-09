def run_kiosk():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    cv2.namedWindow("ERP Lab Kiosk", cv2.WINDOW_NORMAL)

    last_trigger_time = 0
    suggestion_data = None
    age_samples = deque(maxlen=AGE_STABLE_FRAMES)
    smile_counter = 0
    last_face_time = time.time()
    deterministic_index = 0
    age_locked = False
    locked_age_bucket = "Unknown"

    # 🕒 FPS tracking
    start_time = time.time()
    frame_counter = 0

    print("Starting ERP Lab Kiosk (press 'q' to quit, 'r' to reset).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame not available.")
                break

            frame_counter += 1  # 🧮 Count frames

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            box_w = int(frame_w * 0.5)
            box_h = int(frame_h * 0.6)
            box_x1 = (frame_w - box_w) // 2
            box_y1 = (frame_h - box_h) // 2
            box_x2 = box_x1 + box_w
            box_y2 = box_y1 + box_h

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) == 0:
                if suggestion_data and (time.time() - last_face_time) > RESET_TIMEOUT_SEC:
                    suggestion_data = None
                    print("Reset: no face detected for timeout.")
                age_samples.clear()
                smile_counter = 0
                age_locked = False
                locked_age_bucket = "Unknown"

                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
                cv2.putText(frame, "Stand inside the RED box and smile", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "No face detected", (50, frame_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("ERP Lab Kiosk", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('r'):
                    suggestion_data = None
                    age_samples.clear()
                    smile_counter = 0
                    age_locked = False
                    locked_age_bucket = "Unknown"
                    print("Manual reset.")
                continue

            last_face_time = time.time()
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            x = max(0, x)
            y = max(0, y)
            w = max(0, w)

            # (Your age and smile detection logic continues here...)

            cv2.imshow("ERP Lab Kiosk", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                suggestion_data = None
                age_samples.clear()
                smile_counter = 0
                age_locked = False
                locked_age_bucket = "Unknown"
                print("Manual reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # 📊 Print average FPS
        elapsed_time = time.time() - start_time
        average_fps = frame_counter / elapsed_time if elapsed_time > 0 else 0
        print(f"\nAverage FPS during session: {average_fps:.2f}")
