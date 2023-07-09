from ultralytics import YOLO

from PIL import Image
import cv2


def makePrediction(image):
    # Modèle (déjà entrainé)
    model = YOLO("../best.pt")

    # Prédiction
    result = model(image)

    # Get result image
    result_array = result[0].plot()
    # Fix BGR2RGB issue
    result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
    # Save image to PIL format
    result_image = Image.fromarray(result_rgb)

    return result_image
