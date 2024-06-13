# CIFAR.py
import torch

from evaluation.evaluate import evaluate_model, draw_train_process
from training.train import train_model


def main():
    net, train_accs, train_loss = train_model()
    torch.save(net.state_dict(), './model.pth')
    train_iters = range(len(train_accs))
    draw_train_process('training', train_iters, train_loss, train_accs, 'training loss', 'training acc')
    evaluate_model('./model.pth')


if __name__ == '__main__':
    main()
