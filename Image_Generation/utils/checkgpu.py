import tensorflow as tf


def check_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU is available")
    else:
        print("GPU is not available")
