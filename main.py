import config
import dataset
import model
import train
import test
# 设置随机种子
torch.manual_seed(config.seed)
# 定义模型
model = model.LoanModel(config.model_params)
# 训练模型
train.train(model, train_loader, config.train_params, config.loss_fn, config.optimizer)
# 测试模型
test_acc = test.test(model, test_loader)
# 保存模型
torch.save(model.state_dict(), config.model_save_path)
# 加载模型
model.load_state_dict(torch.load(config.model_save_path))
# 重新测试模型
test_acc = test.test(model, test_loader)
if __name__ == '__main__':
    # 训练和测试
    main()