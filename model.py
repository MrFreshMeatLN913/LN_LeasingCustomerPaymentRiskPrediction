import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class LoanModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config['num_inputs'], config['num_hidden1'])
        self.fc2 = nn.Linear(config['num_hidden1'], config['num_hidden2'])
        self.fc3 = nn.Linear(config['num_hidden2'], config['num_outputs'])

    def forward(self, x):
        # 第一隐层
        x = self.fc1(x)
        x = F.relu(x)
        # 第二隐层
        x = self.fc2(x)
        x = F.relu(x)
        # 输出层
        x = self.fc3(x)
        return x

    def validate(self, val_loader):
        self.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(config['device'])
                labels = labels.to(config['device'])
                outputs = self(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                val_correct += predictions.eq(labels).sum().item()

        return val_loss / len(val_loader.dataset), val_correct / len(val_loader.dataset)

    def adjust_learning_rate(optimizer, epoch):
        """调整学习率"""
        lr = config['lr']
        if epoch > config['lr_epoch1']:
            lr = lr * config['lr_decay']
        if epoch > config['lr_epoch2']:
            lr = lr * config['lr_decay']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == '__main__':
    # 测试模型
    inputs = torch.randn(64, 12)
    model = LoanModel(config.model_params)
    outputs = model(inputs)
    print(outputs.shape)
