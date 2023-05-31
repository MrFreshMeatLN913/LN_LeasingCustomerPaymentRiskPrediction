import pandas as pd
from torch.utils.data import Dataset


class LoanData(Dataset):
    def __init__(self, config):
        self.data = pd.read_csv(config.dataset_params['data_path'])
        # 标签
        self.label = self.data[config.label_cols]
        # 数值型输入
        self.num_inputs = self.data[config.num_cols]
        # 分类型输入
        self.cat_inputs = self.data[config.cat_cols]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 数值型输入
        num_inputs = self.num_inputs.iloc[index]
        # 分类型输入
        cat_inputs = self.cat_inputs.iloc[index]
        # 合并输入
        inputs = torch.cat((num_inputs, cat_inputs), 0)
        # 标签
        label = self.label.iloc[index]
        return inputs, label


# 创建训练集和测试集
train_set = LoanData(config, dataset_params['train_data_path'])
test_set = LoanData(config, dataset_params['test_data_path'])

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=config.train_params['batch_size'],
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=config.train_params['batch_size'],
                                          shuffle=False)
