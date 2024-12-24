import imgaug.augmenters as iaa
import cv2
import os
import numpy as np
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

def resize_with_padding(image, target_size=(160, 160)):
    target_width, target_height = target_size
    # h, w, _ = image.shape
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else: 
        h, w = image.shape
    scale = min(target_width / w, target_height / h)
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_image = cv2.resize(image, (new_width, new_height))

    top_pad = (target_height - new_height) // 2
    bottom_pad = target_height - new_height - top_pad
    left_pad = (target_width - new_width) // 2
    right_pad = target_width - new_width - left_pad

    padded_image = cv2.copyMakeBorder(
        resized_image,
        top_pad, bottom_pad, left_pad, right_pad,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0) 
    )

    return padded_image




augmentation_pipeline = iaa.Sequential([
    # iaa.Affine(shear=(-20, 20)),
    iaa.Add((-50, 30)),
    iaa.Fliplr(0.5),
    # iaa.Affine(rotate=(-15, 15)),
    iaa.GaussianBlur(sigma=(0.0, 1.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.075 * 255)),
    iaa.ContrastNormalization((0.8, 1.7)) 
])

def augment_and_save_images(input_dir, output_dir):
    model = YOLO(r"./Models/yolo_model.pt")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for person_name in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_name)
        output_person_path = os.path.join(output_dir, person_name)

        if not os.path.exists(output_person_path):
            os.makedirs(output_person_path)

        for i, image_name in enumerate(os.listdir(person_path)):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img = Image.open(image_path)
            # YOLO Face Detection
            output = model(img)
            results = Detections.from_ultralytics(output[0])

            for bbox, confidence in zip(results.xyxy, results.confidence):
                x1, y1, x2, y2 = map(int, bbox)

                # Crop the face region
                face = image[y1:y2, x1:x2]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                resized_face = resize_with_padding(face_gray, target_size=(64, 64))

                # Save the original face
                original_save_path = os.path.join(output_person_path, f"original_{i}.png")
                cv2.imwrite(original_save_path, resized_face)

                # Expand dimensions for augmentation pipeline
                face_expanded = np.expand_dims(resized_face, axis=-1)
                augmented_images = augmentation_pipeline(images=[face_expanded] * 10)

                # Save augmented images
                for j, aug_img in enumerate(augmented_images):
                    save_path = os.path.join(output_person_path, f"aug_{i}_{j}.png")
                    cv2.imwrite(save_path, aug_img[:, :, 0])

    print(f"Images saved in {output_dir}")


input_dir = r"./database"
output_dir = r"./augmented_database"
augment_and_save_images(input_dir, output_dir)
