from ultralytics import YOLO

import cv2
import os

from ultralytics import YOLO
from django.conf import settings
def infer():
    # Load the YOLO model
    model = YOLO("yolo11n-pose.pt")

    # Open the video file
    video_path = os.path.join(settings.MEDIA_ROOT, 'Test-Footage.mp4')
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))

    writer = cv2.VideoWriter(os.path.join(settings.MEDIA_ROOT, 'Result-Footage.mp4'), cv2.VideoWriter_fourcc(*"AVC1"), frame_rate, (frame_width, frame_height))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            try:
                results = model(frame, device=0)
            except:
                results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)
            writer.write(annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    writer.release()
    cv2.destroyAllWindows()