# 工具函数
# pandas


import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # 测试函数
    model = nn.Linear(10, 5)
    print(count_parameters(model))  # 输出55
seed_everything()  # 设置随机种子
device = get_device()  # 获取设备
print(device)
