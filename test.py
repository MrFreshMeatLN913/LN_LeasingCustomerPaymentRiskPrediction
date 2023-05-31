import config


def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.train_params['device'])
            labels = labels.to(config.train_params['device'])
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct += predictions.eq(labels).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f'Accuracy: {acc}')
    return acc


if __name__ == '__main__':
    # 测试模型
    model = torch.load(config.model_save_path)
    test_acc = test(model, test_loader)
