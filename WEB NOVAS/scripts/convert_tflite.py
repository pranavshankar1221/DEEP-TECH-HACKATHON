import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("models/tf_cnn_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("models/cnn_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created")
