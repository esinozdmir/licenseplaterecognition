import tensorflow as tf
from detection_model import create_detection_model

def load_dataset(tfrecord_path, batch_size=32):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    def _parse_function(example_proto):
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32),
            'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32),
            'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32),
            'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32),
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.image.decode_jpeg(parsed_features['image/encoded'])
        image = tf.image.resize(image, (224, 224)) / 255.0
        bbox = [parsed_features['image/object/bbox/xmin'],
                parsed_features['image/object/bbox/ymin'],
                parsed_features['image/object/bbox/xmax'],
                parsed_features['image/object/bbox/ymax']]
        return image, bbox

    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset.batch(batch_size)

if __name__ == "__main__":
    train_dataset = load_dataset("train.tfrecord")
    test_dataset = load_dataset("test.tfrecord")

    model = create_detection_model()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(train_dataset, epochs=10, validation_data=test_dataset)
    model.save("license_plate_detector.h5")
    print("Model eğitimi tamamlandı ve kaydedildi.")
