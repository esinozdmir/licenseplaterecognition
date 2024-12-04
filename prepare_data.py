import tensorflow as tf
import pandas as pd
import os

def create_tfrecord(output_path, dataset, image_dir):
    with tf.io.TFRecordWriter(output_path) as writer:
        for _, row in dataset.iterrows():
            img_path = os.path.join(image_dir, row['filename'])
            with tf.io.gfile.GFile(img_path, 'rb') as fid:
                encoded_image = fid.read()
            height, width = row['height'], row['width']
            xmin = row['xmin'] / width
            ymin = row['ymin'] / height
            xmax = row['xmax'] / width
            ymax = row['ymax'] / height
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
                'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=[xmin])),
                'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=[ymin])),
                'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=[xmax])),
                'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=[ymax])),
                'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),  # Licence
            }))
            writer.write(tf_example.SerializeToString())

if __name__ == "__main__":
    labels = pd.read_csv("labels.csv")
    train_set = labels.sample(frac=0.8, random_state=42)
    test_set = labels.drop(train_set.index)

    create_tfrecord("train.tfrecord", train_set, "./images")
    create_tfrecord("test.tfrecord", test_set, "./images")

    print("TFRecord dosyaları başarıyla oluşturuldu.")
