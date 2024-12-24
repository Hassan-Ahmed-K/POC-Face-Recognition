from datetime import datetime
import cv2
import pickle
import os
import numpy as np
import shutil
import json

# CONFIDENCE_THRESHOLD = 175

def load_training_data(data_dir):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        
        if os.path.isdir(person_path):
            label_dict[current_label] = person_name
            
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                try:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        # Preprocessing
                        image = resize_with_padding(image, target_size=(64, 64))
                        image = cv2.equalizeHist(image)
                        images.append(np.array(image, dtype=np.uint8))  # Ensure uint8
                        labels.append(current_label)
                    else:
                        print(f"Warning: Image {image_path} could not be read.")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
            
                

                # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # if image is not None:
                #     image = resize_with_padding(image, target_size=(64, 64))
                #     image = cv2.equalizeHist(image)
                #     image = image/255.0
                    
                #     images.append(image)
                #     labels.append(current_label)
            
            current_label += 1
            
    return images, np.array(labels), label_dict

def train_lbph_model(data_dir):
    images, labels, label_dict = load_training_data(data_dir)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    
    recognizer.train(images, labels)
    print("Model trained successfully!")
    
    return recognizer, label_dict

# def save_model_and_labels(recognizer, label_dict, model_path, label_dict_path):
#     try:
#         # Save the label dictionary as JSON
#         with open(label_dict_path, "w") as file:
#             json.dump(label_dict, file, indent=4)
#         print(f"Label dictionary saved to: {label_dict_path}")

#         # Save the LBPH model
#         recognizer.write(model_path)
#         print(f"Model saved to: {model_path}")

#     except Exception as e:
#         print(f"Error saving model or labels: {e}")

def save_model_and_labels(recognizer, label_dict, model_path, label_dict_path):
    
    recognizer.write(model_path)
    with open(label_dict_path, "wb") as file:
        pickle.dump(label_dict, file)
        # json.dump(label_dict, file, indent=4)
    print(f"Model saved to: {model_path}")
    print(f"Label dictionary saved to: {label_dict_path}")

def load_model_and_labels(model_path, label_dict_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    with open(label_dict_path, "rb") as file: 
        label_dict = pickle.load(file)
    print(f"Model loaded from: {model_path}")
    print(f"Label dictionary loaded from: {label_dict_path}")
    return recognizer, label_dict


def resize_with_padding(image, target_size=(64, 64)):
    target_width, target_height = target_size
    h, w = image.shape[:2]
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
        value=(0, 0, 0)  # Black padding
    )
    return padded_image


def save_recognized_image(img, recognized_name,frame_no,timestamp, output_dir,vedio_name):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        person_dir = os.path.join(output_dir, recognized_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)


        filename = os.path.join(person_dir, f'VedioName_{vedio_name}_{recognized_name}_FrameNo-{frame_no}_Time_{timestamp}.jpg')
        # filename = os.path.join(person_dir, f'{recognized_name}_Confidence_{confidence}.jpg')
        # print("output_dir = ", output_dir)
        # print("person_dir = ", person_dir)
        # print("File Name = ", filename)
        # input()
        cv2.imwrite(filename,img)


def person_recognizer(recognizer, image, label_dict, CONFIDENCE_THRESHOLD, frame_no,timestamp,output_dir,vedio_name):
    try:
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grey_image = cv2.equalizeHist(grey_image)
        grey_image = resize_with_padding(grey_image, target_size=(64,64))
        grey_image = grey_image/255.0
        label, confidence = recognizer.predict(grey_image)
        if confidence < CONFIDENCE_THRESHOLD:
            name = label_dict.get(label, "Unknown")
            # print(f"Recognized: {name}, Confidence: {confidence}")
        else:
            name = "Unknown"

        save_recognized_image(image, name,frame_no,timestamp, output_dir,vedio_name)

        return name, confidence
    except Exception as e:
        return None, None

# CONFIDENCE_THRESHOLD = 175

# def move_recognized_image(image_path, recognized_name, confidence, output_dir):

#     if confidence < CONFIDENCE_THRESHOLD:
#         person_dir = os.path.join(output_dir, recognized_name)
#         if not os.path.exists(person_dir):
#             os.makedirs(person_dir)

#         image_name = os.path.basename(image_path)
#         destination_path = os.path.join(person_dir, image_name)
        
#         shutil.move(image_path, destination_path)
#         print(f"Image moved to: {destination_path}")
#     else:
#         print(f"Confidence too high ({confidence}), image not moved.")

def process_test_images(test_dir, recognizer, output_dir):

    for image_name in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_name)
        
        # Ensure the file is an image
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            name, confidence = input_image_embeddings(recognizer, image_path)
            if name is not None and confidence is not None:
                print(f"Image: {image_name}, Name: {name}, Confidence: {confidence}")


def input_image_embeddings(recognizer, image_path, label_dict):
    try:
        test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if test_image is None:
            raise ValueError("Image not found or invalid image path.")
        
        test_image = resize_with_padding(test_image, target_size=(64, 64))
        print("===============================================")
        
        label, confidence = recognizer.predict(test_image)
        name = label_dict.get(label, "Unknown")
        
        return name, confidence
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

# data_dir = r"./augmented_database"

# recognizer, label_dict = train_lbph_model(data_dir)
# lbph_model_path = r"./Models/lbph_model.yaml" 
# label_dict_path = r"./Models/label_dict.pkl"

# print("recognizer = ", recognizer)
# print("label_dict = ", label_dict)


# save_model_and_labels(recognizer, label_dict, lbph_model_path, label_dict_path)