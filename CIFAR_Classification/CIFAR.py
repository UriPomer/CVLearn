import sys

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot

from CIFAR_Classification.evaluation import run_example
from cnn import define_model
from data_loader import load_dataset, prep_pixels

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU instead")


class TrainingProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch + 1}: loss = {logs['loss']:.4f}, accuracy = {logs['accuracy']:.4f}, val_loss = {logs['val_loss']:.4f}, val_accuracy = {logs['val_accuracy']:.4f}")


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


def main():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model()
    progress = TrainingProgress()
    history = model.fit(trainX, trainY, epochs=100, batch_size=64, verbose=1, validation_data=(testX, testY),
                        callbacks=[progress])
    model.save('./model/final_model.h5')
    summarize_diagnostics(history)


if __name__ == '__main__':
    # main()
    # evaluate_model()
    run_example()
