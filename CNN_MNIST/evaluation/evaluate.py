# evaluation/evaluate.py
import matplotlib.pyplot as plt
import torch

from CNN_MNIST.CNN import CNN
from CNN_MNIST.utils.data_loader import get_data_loaders


def draw_train_process(title, iters, costs, accs, label_cost, label_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("acc(%)", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=label_acc)
    plt.legend()
    plt.grid()
    plt.show()


def evaluate_model(model_path):
    _, test_loader = get_data_loaders()
    test_net = CNN()
    test_net.load_state_dict(torch.load(model_path))
    test_net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = test_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = test_net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %d : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))
    draw_train_process('test', range(10), class_total, class_correct, 'total', 'correct')


def main():
    model_path = './model.pth'
    evaluate_model(model_path)


if __name__ == '__main__':
    main()
