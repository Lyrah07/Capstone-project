from ultralytics import YOLO

import cv2
import os
import platform
import uuid

from ultralytics import YOLO
from django.conf import settings
from pandas import pandas, DataFrame

def infer() -> None: 
    # Load the YOLO model
    model = YOLO("yolo11n-pose.pt")

    # Open the video file
    video_path = os.path.join(settings.MEDIA_ROOT, 'Test-Footage.mp4')
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))

    fragment_duration = "5"

    output_fourcc = cv2.VideoWriter_fourcc(*'X264')
    output_path_fragment = os.path.join(settings.MEDIA_ROOT, "segment_%05d.mp4")
    output_path_playlist = os.path.join(settings.MEDIA_ROOT, "index.m3u8")
    
    if(platform.system() == "Windows"):
        output_path_fragment.replace("\\", "/")
        output_path_playlist.replace("\\", "/")

    gst_string = (
        "appsrc ! videoconvert ! nvh264enc tune=zerolatency bitrate=21516 speed-preset=superfast ! "
        "mpegtsmux ! hlssink "
        f"location={output_path_fragment} "
        f"playlist-location={output_path_playlist} "
        f"target-duration={fragment_duration} "
    )

    print(
        f"Four CC: {output_fourcc}",
        f"Output Path Fragment: {output_path_fragment} ",
        f"Output Path Playlist: {output_path_playlist} ",
        f"Gstream String: {gst_string} ",
        sep="\n"
    )

    writer = cv2.VideoWriter(gst_string, cv2.CAP_GSTREAMER, output_fourcc, frame_rate, (frame_width, frame_height))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            try:
                results = model(frame, device=0, verbose=False)
            except:
                results = model(frame)

            # Get the Objects in frame.
            objects_in_frame : Results = results[0].boxes.data

            object_data_frame = pandas.DataFrame(objects_in_frame.cpu()).astype('float')


            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            checkFor(object_data_frame, [], frame)

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

def crop(coords: [int, int, int, int], frame: cv2.UMat, path: str = "") -> None :
    file_name : str = f"detected_person_{uuid.uuid4()}"

    path = os.path.join(settings.MEDIA_ROOT, "screenshots")

    y1, y2, x1, x2 = coords
    cropped_image = frame[y1:y2, x1:x2]
    file_path = os.path.join(path, f"{file_name}.png", )
    file_path.replace("\\", "/")

    if not cv2.imwrite(file_path, cropped_image) :
        raise Exception(f'Unable to save Image at {file_path}')
    print(f"Should have saved at: {file_path}")

def checkFor(data_frame: DataFrame, object_list: list, frame: cv2.UMat) -> None : 
    for index, row in data_frame.iterrows():
        x1, y1, x2, y2, classification = skimDataFromDF(row)
        infered_object : [int, int, int, int] = [x1, y1, x2, y2]
        # crop([y1,y2,x1,x2], frame)

def skimDataFromDF(row) -> list:
    x1 = int(row[0])
    y1 = int(row[1])
    x2 = int(row[2])
    y2 = int(row[3])
    d = int(row[5])

    return [x1, y1, x2, y2, d]