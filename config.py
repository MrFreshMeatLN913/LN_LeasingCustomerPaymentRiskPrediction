

model_params = {
        'num_inputs':12,    #输入节点数
        'num_hidden1':64,   #第一隐层节点数
        'num_hidden2':32,   #第二隐层节点数
        'num_outputs':2     #输出节点数
    }

train_params = {
        'batch_size':64,    #批大小
        'lr':0.01,          #学习率
        'num_epoch':10,     #训练次数
        'device':'cuda'     #使用GPU
}

dataset_params = {
        'train_data_path':'train_data.csv',             #训练数据路径
        'test_data_path':'test_data.csv'                #测试数据路径
}
model_save_path = 'model.pth'                           #模型保存路径
num_cols = ['月收入', '金融总额', '融资期限']               #数值型特征
cat_cols = ['婚姻状况', '所在行业', '房产情况']             #分类特征
label_cols = ['客户ID','是否超期']                        #标签
seed = 42                                               #随机种子
log_interval = 100                                      #记录间隔