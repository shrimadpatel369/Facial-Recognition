import cv2
import numpy as np
import os

# Path to the dataset
dataset_path = 'E:/AU/mat/dataset'

# Load images and labels
images = []
labels = []
label_dict = {}
current_label = 0

# Define the desired image size (e.g., 100x100)
image_size = (100, 100)

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        label_dict[current_label] = person_name
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    # Normalize the image
                    image = cv2.equalizeHist(image)
                    # Resize the image to the desired size
                    image = cv2.resize(image, image_size)
                    images.append(image)
                    labels.append(current_label)
            except Exception as e:
                print(f"Error reading {image_path}: {e}")
        current_label += 1

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Create the Eigenface model
model = cv2.face.EigenFaceRecognizer_create()

# Train the model
model.train(images, labels)

# Function to recognize a face
def recognize_face(image_path, threshold=4000):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Normalize the image
            image = cv2.equalizeHist(image)
            # Resize the image to the desired size
            image = cv2.resize(image, image_size)
            label, confidence = model.predict(image)
            if confidence < threshold:
                return label_dict[label], confidence
            else:
                return "Unknown", confidence
        else:
            return None, None
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None, None

# Test the recognizer
test_image_path = 'E:/AU/mat/7.jpg'
person_name, confidence = recognize_face(test_image_path)
if person_name is not None:
    print(f'Recognized as {person_name} with confidence {confidence}')
else:
    print('Face not recognized')

# Function to recognize faces in a group image
def recognize_faces_in_group(image_path, threshold=4000):
    try:
        # Load the group image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load the face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        results = []
        
        for (x, y, w, h) in faces:
            # Extract the face region
            face = gray[y:y+h, x:x+w]
            
            # Normalize the face
            face = cv2.equalizeHist(face)
            
            # Resize the face to the desired size
            face_resized = cv2.resize(face, image_size)
            
            # Recognize the face
            label, confidence = model.predict(face_resized)
            if confidence < threshold:
                person_name = label_dict[label]
            else:
                person_name = "Unknown"
            
            # Append the result
            results.append((person_name, confidence, (x, y, w, h)))
        
        return results
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

# Test the recognizer on a group image
group_image_path = 'E:/Photos/bakrol-vadtal trip/5870771173653071583.jpg'
results = recognize_faces_in_group(group_image_path)

# Print the results
for person_name, confidence, (x, y, w, h) in results:
    print(f'Recognized {person_name} with confidence {confidence} at location {(x, y, w, h)}')