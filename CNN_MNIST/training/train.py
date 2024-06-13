# training/train.py
import torch
import torch.optim as optim

from CNN_MNIST.CNN import CNN
from CNN_MNIST.utils.data_loader import get_data_loaders


def train_model(num_epochs=3):
    train_loader, _ = get_data_loaders()
    net = CNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), learning_rate=0.001, momentum=0.9)

    train_accs = []
    train_loss = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            train_loss.append(loss.item())

            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            train_accs.append(100 * correct / total)

    print('Finished Training')
    return net, train_accs, train_loss


if __name__ == '__main__':
    trained_model, train_accs, train_loss = train_model()
    torch.save(trained_model.state_dict(), './model.pth')
