from torch import nn

import config
import torch
import model

from dataset import train_loader
from model import LoanModel


def train(model, train_loader, config, loss_fn, optimizer, val_loader=None):
    model.train()
    train_loss = 0
    correct = 0
    for epoch in range(config['num_epoch']):
        for inputs, labels in train_loader:
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = loss_fn(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            correct += predictions.eq(labels).sum().item()

        # 验证
        val_loss, val_acc = model.validate(model, val_loader)

        # 调整学习率
        model.adjust_learning_rate(optimizer, epoch)

        # 打印训练信息
        if epoch % config['log_interval'] == 0:
            print(
                f'Epoch: {epoch + 1} Train loss: {train_loss / len(train_loader.dataset):.4f} Acc: {correct / len(train_loader.dataset):.4f}')
            print(f'Val loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

    return train_loss / len(train_loader.dataset)


if __name__ == '__main__':
    model = LoanModel(config.model_params)
    model.to(config.train_params['device'])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.train_params['lr'])
    train_loss = train(model, train_loader, config.train_params, loss_fn, optimizer)
    print(f'Train loss: {train_loss:.4f}')
