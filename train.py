# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 19:36
# @Author  : chenlelan
# @File    : train.py


import logging
import os
from datetime import datetime
from model.DySAT.models import DySAT
from utils.preprocess import *
from utils.minibatch import *
from utils.write_to_csv import *
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchsummary import summary
import numpy as np
from config import DefaultConfig
from torch.utils.tensorboard import SummaryWriter

cfg = DefaultConfig
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

# 输入正确的基本模型和模型名称，以获取解析器文件夹，从中加载cfg
output_dir = "./logs/{}/".format(cfg.base_model)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
output_dir = output_dir + "binary_class/"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# 设置输出子文件路径
LOG_DIR = output_dir + cfg.logs_dir
SAVE_DIR = output_dir + cfg.save_dir
CSV_DIR = output_dir + cfg.csv_dir
MODEL_DIR = output_dir + cfg.model_dir

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

if not os.path.isdir(CSV_DIR):
    os.mkdir(CSV_DIR)

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)


datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()
# 训练日志记录格式：为每次运行创建一个新的日志目录。目录的默认名称是“logs”，而<>.log的内容将每天附加 => 每天一个日志文件。
log_file = LOG_DIR + '/%s_%s_%s_%s.logs' % (cfg.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day))

log_level = logging.INFO
logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

# 遍历 DefaultConfig 类的所有属性和值，并将它们保存到日志中
for attr, value in DefaultConfig.__dict__.items():
    if not attr.startswith('__') and not callable(value):
        logging.info(f'{attr}: {value}')

# 创建结果日志csv文件
output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (cfg.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day))

# 加载数据
graphs, adjs, label, classes = load_graphs(cfg.train_dataset)
feats = load_feats(cfg.train_dataset)
eval_graphs, eval_adjs, eval_label, eval_classes = load_graphs(cfg.eval_dataset)
eval_feats = load_feats(cfg.eval_dataset)
num_features = len(feats[0][0])
print("node_features_dim: ", num_features)

minibatchIterator = MinibatchIterator(graphs, feats, adjs, label, classes,
                                          num_features, batch_size=cfg.batch_size)
evalbatchIterator = MinibatchIterator(eval_graphs, eval_feats, eval_adjs, eval_label, eval_classes,
                                          num_features, batch_size=cfg.batch_size)

model = DySAT(num_features, training=True)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU 可用")
else:
    device = torch.device("cpu")
    print("GPU 不可用，将使用 CPU")
model.to(device)

# Result accumulators.
epochs_auc_result = {}
epochs_val_score = {}
epochs_train_loss = []


for epoch in range(cfg.epochs):
    minibatchIterator.shuffle()
    # prev_embed用于保存之前batch中获得的包含邻居信息的节点嵌入（即历史信息）
    prev_embed = []
    prev_node = []
    eval_prev_embed = []
    eval_prev_node = []
    epoch_loss = 0.0
    it = 0
    print('Epoch: %04d' % (epoch))
    batch = 0
    while not minibatchIterator.end():
        # Construct feed dictionary
        feed_dict = minibatchIterator.next_minibatch_feed_dict()
        feed_dict.update({'spatial_drop': cfg.spatial_drop})
        feed_dict.update({'temporal_drop': cfg.temporal_drop})
        # Training step
        trainable_params = model.parameters()

        with torch.enable_grad():
            # 1：在上下文中执行前向传播操作
            y, embed = model(feed_dict)
            train_cost, class_cost, graph_cost, reg_cost = model._loss()

        # 2:在 with 上下文之后计算梯度
        train_cost.backward()
        # 遍历模型的可训练参数,打印梯度
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Parameter: {name}, Gradient: {param.grad}")
        # 3:根据给定的 maximum_gradient_norm 对梯度进行裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_gradient_norm)
        # 4:创建 Adam 优化器
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        # 5:更新梯度
        optimizer.step()

        # Print results
        logging.info("Mini batch Iter: {} train_loss= {:.5f}".format(it, train_cost))
        logging.info("Mini batch Iter: {} class_cost= {:.5f}".format(it, class_cost))
        logging.info("Mini batch Iter: {} graph_cost= {:.5f}".format(it, graph_cost))
        logging.info("Mini batch Iter: {} reg_cost= {:.5f}".format(it, reg_cost))
        epoch_loss += train_cost
        it = it + 1
        batch = batch + 1
    epoch_loss = epoch_loss / it
    epochs_train_loss.append(epoch_loss)
    logging.info("Mean Loss at epoch {} : {:.5f}".format(epoch, epoch_loss))
    print("Mean Loss at epoch {} : {}".format(epoch, epoch_loss))

    # eval_data
    if epoch % cfg.eval_freq == 0:
        evalbatchIterator.shuffle()
        val_scores = []
        i = 1
        while not evalbatchIterator.end():
            print('****** eval_data batch {} ******'.format(i))
            eval_feed_dict = evalbatchIterator.next_minibatch_feed_dict()
            eval_feed_dict.update({'spatial_drop': 0.0})
            eval_feed_dict.update({'temporal_drop': 0.0})
            y, eval_embed = model(eval_feed_dict)

            eval_scores = model._result_score()
            print('result:', eval_scores)
            val_scores.append(eval_scores)
            i = i + 1
        print('epoch_result', val_scores)
        val_scores = torch.mean(torch.Tensor(val_scores), dim=0).numpy()
        print('epoch reduce mean result:', val_scores, type(val_scores), val_scores[3])
        epochs_val_score[epoch] = val_scores
        auc_val = val_scores[3]  # AUC
        epochs_auc_result[epoch] = auc_val

        # 保存模型
        if (epoch == 0) or (epoch > 0 and auc_val >= max(epochs_auc_result.values())):
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            save_path = MODEL_DIR + "/" + "model_{}".format(cfg.dataset) + ".pth"
            torch.save(model.state_dict(), save_path)
        # 释放缓存分配器中当前持有的所有未占用的缓存内存
        torch.cuda.empty_cache()

# 记录最佳epoch的训练log
best_epoch = max(epochs_auc_result, key=epochs_auc_result.get)
print("Best epoch ", best_epoch)
logging.info("Best epoch {}".format(best_epoch))

best_val_result = epochs_val_score[best_epoch]

print("Best epoch {}, Val acc {}".format(best_epoch, best_val_result[0]))
print("Best epoch {}, Val recall {}".format(best_epoch, best_val_result[1]))
print("Best epoch {}, Val precision {}".format(best_epoch, best_val_result[2]))
print("Best epoch {}, Val AUC {}".format(best_epoch, best_val_result[3]))
print("Best epoch {}, Val f1_score {}".format(best_epoch, best_val_result[4]))
# print("Best epoch {}, Val FPR {}".format(best_epoch, best_val_result[5]))

logging.info("Best epoch val results acc: {}, recall: {}, precision: {}, AUC: {}, f1_score: {}\n".format(
    best_epoch, best_val_result[0], best_val_result[1], best_val_result[2], best_val_result[3], best_val_result[4]))

write_eval_result(best_val_result, output_file, cfg.base_model, cfg.dataset, mod='eval_data')
write_epoch_loss(epochs_train_loss, output_file, cfg.base_model, cfg.dataset, mod='train_data')

print(model)
# 绘制loss曲线
plt.figure()
plt.plot(list(range(1, len(epochs_train_loss) + 1)), epochs_train_loss)
plt.title('Train loss of each epochs')
plt.xlabel('epoch')
plt.ylabel('train_data loss')
plt.show()
