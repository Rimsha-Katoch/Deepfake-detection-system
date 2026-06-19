import cv2
import os


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_faces(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_filename = os.path.join(output_folder, f"face_{count}.jpg")
            cv2.imwrite(face_filename, face)
            count += 1

    print(f"Extracted {count} faces.")