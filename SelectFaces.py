from mtcnn import MTCNN
import cv2
import os

def detect_and_crop_faces(image_path, output_folder, image_number, target_size=(224, 224)):
    # Charger l'image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Initialiser le détecteur MTCNN
    detector = MTCNN()

    # Détecter les visages
    faces = detector.detect_faces(image)

    if not faces:
        print(f"Aucun visage détecté dans l'image {image_path}")

    # Extraire et sauvegarder chaque visage détecté
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        x, y = abs(x), abs(y)

        # Extraire le visage
        face_image = image[y:y+height, x:x+width]

        # Redimensionner le visage à la taille cible
        face_image = cv2.resize(face_image, target_size)

        # Sauvegarder l'image avec une bonne qualité
        output_path = os.path.join(output_folder, f'face_{i}_image_{image_number}.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        print(f"Visage détecté et sauvegardé dans {output_path}")

def detect_and_crop_faces_in_folder(folder, output_folder, target_size=(224, 224)):
    print("Détection lancée")

    # Assurez-vous que le dossier de sortie existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    i = 0

    # Parcourez les fichiers dans le dossier
    for image in os.listdir(folder):
        i += 1
        if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.JPG') or image.endswith('.PNG'):
            image_path = os.path.join(folder, image)
            detect_and_crop_faces(image_path, output_folder, i, target_size)

    print(f"Faces détectées et rognées dans {output_folder}")

# Appel de la fonction
detect_and_crop_faces_in_folder("Labeled_data/RawData", "Labeled_data/CroppedFaces", target_size=(224, 224))