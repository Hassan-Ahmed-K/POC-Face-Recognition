import cv2
import os
from PIL import Image
from ultralytics import YOLO
from supervision.detection.core import Detections
import shutil
from LBPH import load_model_and_labels,person_recognizer



CONFIDENCE_THRESHOLD = 169.1

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "augmented_database")
output_dir = os.path.join(base_dir, "recognized_faces")
lbph_model_path = os.path.join(base_dir, "Models", "lbph_model.yaml")
label_dict_path = os.path.join(base_dir, "Models", "label_dict.pkl")
yolo_model_path = os.path.join(base_dir, "Models", "yolo_model.pt")


# data_dir = r"./augmented_database"
# output_dir = r"./recognized_faces"
# lbph_model_path = r"./Models/lbph_model.yaml" 
# label_dict_path = r"./Models/label_dict.pkl"


def processed_vedios(src_dir,destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    try:
        for vedio in os.listdir(src_dir):
            print("===============================================")
            print(f"Start Processing vedio: {vedio}")
            print("===============================================")
            vedio_path = os.path.join(src_dir, vedio)
            face_recognition(model_path=yolo_model_path,lbph_model_path=lbph_model_path,label_dict_path=label_dict_path,video_path=vedio_path,output_dir=output_dir,vedio_name=vedio)
            destination_path = os.path.join(destination_dir, vedio)
            shutil.move(vedio_path, destination_path)

    except Exception as e:
        print(f"Error Processing vedio: {e}")

def face_recognition(model_path,lbph_model_path,label_dict_path,video_path,output_dir,vedio_name,threshold=0.5):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

    frame_no = 0
    recognizer, label_dict = load_model_and_labels(lbph_model_path, label_dict_path)
    print("Loaded label dictionary:", label_dict)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_seconds = frame_no / fps
        timestamp_formatted = "{:02}-{:02}-{:02}".format(
            int(timestamp_seconds // 3600), 
            int((timestamp_seconds % 3600) // 60), 
            int(timestamp_seconds % 60)
        )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        output = model(pil_frame)
        results = Detections.from_ultralytics(output[0])

        for bbox, confidence in zip(results.xyxy, results.confidence):
            if confidence >= threshold:
                x1, y1, x2, y2 = map(int, bbox)

                # padding_ratio = 0.5
                # width_padding = int((x2 - x1) * padding_ratio)
                # height_padding = int((y2 - y1) * padding_ratio)

                # x1_padded = max(0, x1 - width_padding)
                # y1_padded = max(0, y1 - height_padding)
                # x2_padded = min(frame_width, x2 + width_padding)
                # y2_padded = min(frame_height, y2 + height_padding)
                cropped_face = frame[y1:y2, x1:x2]

                name, confidence = person_recognizer(recognizer, cropped_face, label_dict, CONFIDENCE_THRESHOLD, frame_no,timestamp_formatted,output_dir,vedio_name)

                cv2.rectangle(frame, (x1,y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, str(name), (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # cv2.putText(
        #     frame,
        #     f"Timestamp: {timestamp_formatted}",
        #     (10, 30),  #
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,  
        #     (0, 255, 0),  
        #     2,  #
        # )

        frame_no += 1
        aspect_ratio = frame_width / frame_height
        display_width = 800
        display_height = int(display_width / aspect_ratio)
        resized_frame = cv2.resize(frame, (display_width, display_height))
        cv2.imshow("Face Detection", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

print("__name__ = ",__name__)

if __name__ == "__main__":

    src_dir = os.path.join(base_dir, "raw vedio")
    destination_dir = os.path.join(base_dir, "processed vedio")

    processed_vedios(src_dir,destination_dir)
