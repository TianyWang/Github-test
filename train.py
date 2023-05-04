import os
import time
import numpy as np
import paddle
import random
from paddle.io import DataLoader
import paddle.nn.functional as F
from DLinear import WPFModel
import logging
import loss as loss_factory
from metrics import metric
import optimization as optim
from common import EarlyStopping
from common import adjust_learning_rate
from utils import _create_if_not_exist
from prepare import prep_env
from datasets import Split_csv
from datasets import WPFDataset


# 数据增强
def data_augment(x, y, p=0.9, alpha=0.5, beta=0.5):
    """
    Regression SMOTE
    """
    # batch+x,batch_y: batch_size, L,C=134
    # input_y: batch_size, C=134, L, feature_len
    batch_size = x.shape[0]
    random_values = paddle.rand([batch_size])
    idx_to_change = random_values < p

    # ensure that first element to switch has probability > 0.5
    # beta(a,b,size)
    np_betas = np.random.beta(alpha, beta, batch_size) / 2 + 0.5
    random_betas_1 = paddle.to_tensor(
        np_betas, dtype="float32").reshape([-1, 1, 1])
    random_betas_2 = paddle.to_tensor(
        np_betas, dtype="float32").reshape([-1, 1, 1, 1])
    index_permute = paddle.randperm(batch_size)

    x[idx_to_change] = random_betas_1[idx_to_change] * x[idx_to_change]
    x[idx_to_change] += (1 - random_betas_1[idx_to_change]) * x[index_permute][idx_to_change]

    y[idx_to_change] = random_betas_1[idx_to_change] * y[idx_to_change]
    y[idx_to_change] += (1 - random_betas_1[idx_to_change]) * y[index_permute][idx_to_change]

    return x, y


# logger
def getLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    work_dir = os.path.join(settings['logs_path'],
                            time.strftime("%Y-%m-%d-%H.%M", time.localtime()))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    return logger


# logger = getLogger()

# 训练验证
# 参数 ： 配置，模型，是否调试
def train_and_val(settings, model, is_debug=False):
    """
    Desc:
        Training and validation
    Args:
        settings:env_settings
        model:model
        is_debug:False
    Returns:
        None
    """
    
    # 读取所有数据记录 | 按照比例划分数据 (默认val比例 0.1)
    train_df, val_df = Split_csv([os.path.join(settings["data_path"], i) for i in os.listdir(settings["data_path"])])
    # 训练数据
    train_dataset = WPFDataset(
        train_df,
        size=[settings["input_len"], settings["label_len"], settings["output_len"]],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
    )
    # 验证数据
    val_dataset = WPFDataset(
        val_df,
        size=[settings["input_len"], settings["label_len"], settings["output_len"]],
        scaler=train_dataset.get_scaler()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        drop_last=True
    )

    # 是否并行
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # 损失函数 使用 MES_loss
    loss_fn = getattr(loss_factory, settings["name"])(
        **dict({"name": "Filter_MSE_Loss"}))

    # 优化器 使用固定学习率
    opt = optim.get_optimizer(model=model, learning_rate=settings["lr"])

    # 已训练batch总数
    global_step = 0

    # 路径中文件夹是否存在 不存在新建
    _create_if_not_exist(settings["checkpoints"])
    path_to_model = settings["checkpoints"]

    # 训练结果保存器
    early_stopping = EarlyStopping(patience=settings["patient"])

    ### 开始训练 ###
    valid_records = []
    epoch_start_time = time.time()
    for epoch in range(settings["train_epochs"]):
        # 训练模式
        model.train()
        # index ，（input，GT）
        for i, (batch_x, batch_y) in enumerate(train_loader):
            # 数据类型转换 | 数据增强
            batch_x = batch_x.astype('float32')
            batch_y = batch_y.astype('float32')
            batch_x, batch_y = data_augment(batch_x, batch_y)
            # 优化器初始化
            opt.clear_gradients()
            # 正向传播
            pred_y = model(batch_x)
            # 计算损失
            loss = loss_fn(pred_y, batch_y)
            # 反向传播
            loss.backward()
            # 参数优化
            opt.step()
            # 训练步数 自增
            global_step += 1
            # 打印损失
            if paddle.distributed.get_rank() == 0 and \
               global_step % settings["log_per_steps"] == 0:
                logger.info("Step %s Train MSE-Loss: %s RMSE-Loss: %s" %
                            (global_step, loss.numpy()[0],
                             (paddle.sqrt(loss)).numpy()[0]))
        # debug模式 输出每个epoch时间
        if is_debug:
            epoch_end_time = time.time()
            print("Elapsed time for epoch-{}: {}".format(epoch, epoch_end_time - epoch_start_time))
            epoch_start_time = epoch_end_time

        # 验证
        if paddle.distributed.get_rank() == 0:
            valid_r = validation(val_loader,train_dataset,
                                val_dataset,model,loss_fn,settings)
            valid_records.append(valid_r)
            logger.info("Valid " + str(dict(valid_r)))
            val_loss = valid_r['turb_score']

            # 结果保存至 path_to_model
            early_stopping(val_loss, model, path_to_model)
            logger.info("the best model's score is:{}".format(early_stopping.val_loss_min))
            
            # 提前结束训练
            if early_stopping.early_stop:
                print("Early stopped! ")
                break

        # 调整学习率
        adjust_learning_rate(opt, epoch + 1, settings)


# 验证
@paddle.no_grad()
def validation(valid_data_loader,
               train_dataset,
               val_dataset,
               model,
               loss_fn,
               settings
               ):
    model.eval()
    losses = []
    pred_batch = []
    gold_batch = []
    scaler = train_dataset.get_scaler()
    step = 0
    for batch_x, batch_y in valid_data_loader:
        # batch_x,batch_y: batch_size,input_len/label+output_len,C(134turb)
        # input_y: batch_size, 134,label+output_len, C(10features)
        batch_x = batch_x.astype('float32')
        batch_y = batch_y.astype('float32')
        # batch_size, 134,output_len
        pred_y = model(batch_x)
        loss = loss_fn(pred_y, batch_y)

        losses.append(loss.numpy()[0])
        inverse_pred_y = scaler.inverse_transform_y(pred_y)
        inverse_truth = scaler.inverse_transform_y(batch_y)

        pred_batch.append(inverse_pred_y.numpy())
        gold_batch.append(inverse_truth.numpy())
        step += 1
        if paddle.distributed.get_rank(
        ) == 0 and step % settings["log_per_steps"] == 0:
            logger.info("Step %s Val MSE-Loss: %s RMSE-Loss: %s" %
                        (step, loss.numpy()[0],
                         (paddle.sqrt(loss)).numpy()[0]))

    # N', 134, output_len
    pred_batch = np.concatenate(pred_batch, axis=0)
    gold_batch = np.concatenate(gold_batch, axis=0)
    # N', 134, output_len, 1
    pred_batch = np.expand_dims(pred_batch, -1)
    gold_batch = np.expand_dims(gold_batch, -1)
    # 134, N', output_len, 1
    pred_batch = np.transpose(pred_batch, [1, 0, 2, 3])
    gold_batch = np.transpose(gold_batch, [1, 0, 2, 3])

    _mae, _rmse = metric(pred_batch, gold_batch)

    output_metric = {
        'farm_mae': _mae,
        'farm_rmse': _rmse,
        'turb_score': (_mae + _rmse) / 2,
        'loss': np.mean(losses),
    }
    # 返回 dict
    return output_metric



# 主函数
if __name__ == "__main__":
    
    # 设置相同的随机数种子
    fix_seed = 3407
    random.seed(fix_seed)
    paddle.seed(fix_seed)
    np.random.seed(fix_seed)

    # 设置 初始化
    settings = prep_env() # 在 prepare.py中
    # 设置 log
    logger = getLogger()

    # 打印配置
    logger.info("The experimental settings are: \n{}".format(str(settings)))
    
    ######## 设置训练环境 ########
    start_train_time = time.time() # 训练开始时间
    logger.info('\n>>>>>>>Start training \n')
    # 实例化模型
    model = WPFModel(settings)
    # 训练
    train_and_val(settings, model=model, is_debug=settings["is_debug"])
    # debug模式 输出训练时间
    if settings["is_debug"]:
        end_time = time.time()
        end_train_time = end_time
        print("\nTotal time in training {} turbines is "
              "{} secs".format(settings["capacity"], end_train_time - start_train_time))
    ############################