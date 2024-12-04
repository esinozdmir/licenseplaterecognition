from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def create_detection_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Pre-trained ağı dondurun
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='sigmoid')  # xmin, ymin, xmax, ymax
    ])
    return model
