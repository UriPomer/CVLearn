import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

from CIFAR_Classification.data_loader import load_dataset, prep_pixels


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(32, 32))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


def evaluate_model():
    # Load dataset
    trainX, trainY, testX, testY = load_dataset()
    # Prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # Load model
    model = load_model('final_model.h5')
    # Evaluate model on test dataset
    results = model.evaluate(testX, testY, verbose=0)
    loss, accuracy = results[0], results[1]
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Predict the classes
    y_pred = model.predict(testX)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(testY, axis=1)

    # Compare predictions with true labels
    correct = (y_pred_classes == y_true)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(correct, 'o', label='Correct')
    plt.xlabel('Sample Index')
    plt.ylabel('Correct Prediction')
    plt.title('Prediction Correctness per Sample')
    plt.legend()
    plt.show()

    # Optionally, you can plot the prediction accuracy for each class
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    for i in range(len(y_true)):
        label = y_true[i]
        class_total[label] += 1
        if correct[i]:
            class_correct[label] += 1

    class_accuracy = class_correct / class_total
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), class_accuracy, color='blue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.show()


# load an image and predict the class
def run_example():
    # load the image
    img = load_image('sample_image.png')
    # load model
    model = load_model('final_model.h5')
    # predict the class
    predictions = np.argmax(model.predict(img), axis=-1)
    print(predictions[0])
