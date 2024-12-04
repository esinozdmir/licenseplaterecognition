import cv2
import pytesseract
import tensorflow as tf

def recognize_license_plate(image_path, model_path):
    model = tf.keras.models.load_model(model_path)
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224)) / 255.0
    bbox = model.predict(tf.expand_dims(resized_image, axis=0))[0]

    h, w, _ = image.shape
    xmin, ymin, xmax, ymax = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)
    cropped_plate = image[ymin:ymax, xmin:xmax]

    text = pytesseract.image_to_string(cropped_plate, config="--psm 7")
    return text

if __name__ == "__main__":
    plate_text = recognize_license_plate("arac.jpg", "license_plate_detector.h5")
    print("Tespit Edilen Plaka:", plate_text)
