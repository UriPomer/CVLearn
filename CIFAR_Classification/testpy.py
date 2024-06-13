from tensorflow.python.client import device_lib

dev = device_lib.list_local_devices()
print(dev)
