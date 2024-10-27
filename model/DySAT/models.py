# encoding: utf-8

from model.DySAT.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Recall, Precision, AUC
from sklearn.metrics import roc_curve, auc
from config import DefaultConfig

cfg = DefaultConfig


# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package


class DySAT(nn.Module):
    def _accuracy(self):
        pass

    def __init__(self, num_features, training=False, **kwargs):
        super(DySAT, self).__init__()
        self.attn_wts_all = []
        self.temporal_attention_layers = nn.ModuleList()
        self.structural_attention_layers = nn.ModuleList()
        self.training = training
        if training:
            self.spatial_drop = cfg.spatial_drop
            self.temporal_drop = cfg.temporal_drop
        else:
            self.spatial_drop = 0.0
            self.temporal_drop = 0.0
        self.num_features = num_features
        self.structural_head_config = cfg.structural_head_config
        self.structural_layer_config = cfg.structural_layer_config
        self.temporal_head_config = cfg.temporal_head_config
        self.temporal_layer_config = cfg.temporal_layer_config
        self.training = training
        self.result_scores = []
        self.labels = None
        self._build()

    def _build(self):
        # 1: Neighbors Attention Layers
        input_dim = self.num_features
        for i in range(0, len(self.structural_head_config)):
            if i > 0:
                input_dim = self.structural_layer_config[i - 1]
                print("structural input dim:", input_dim)
            self.structural_attention_layers.append(StructuralAttentionLayer(input_dim=input_dim,
                                                                             output_dim=self.structural_layer_config[i],
                                                                             n_heads=self.structural_head_config[i],
                                                                             attn_drop=self.spatial_drop,
                                                                             ffd_drop=self.spatial_drop,
                                                                             sparse_inputs=False,
                                                                             residual=cfg.use_residual))

        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]
        for i in range(0, len(self.temporal_layer_config)):
            if i > 0:
                input_dim = self.temporal_layer_config[i - 1]
            temporal_layer = TemporalAttentionLayer(input_dim=input_dim, n_heads=self.temporal_head_config[i],
                                                    attn_drop=self.temporal_drop, num_time_steps=cfg.window,
                                                    residual=False)
            self.temporal_attention_layers.append(temporal_layer)

        # 3:创建全连接层，实现二元分类
        self.embed_layer = nn.Linear(self.temporal_layer_config[-1], 8)
        # self.fc_layer = nn.Linear(self.temporal_layer_config[-1], 1)
        self.fc_layer = nn.Linear(8, 1)


    def forward(self, placeholders):
        self.placeholders = placeholders
        self.num_time_steps = len(placeholders['features'].numpy())
        # print('input feature: ', placeholders['features'])
        # print('num time steps: ', self.num_time_steps)
        # 1: Neighbors Attention forward
        input_list = placeholders['features']  # List of centre node feature matrices. [T, N_t, S]
        adjs = self.placeholders['adjs']  # [T, N_t, N_t]
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        input_list.to(device)
        adjs.to(device)
        for idx, layer in enumerate(self.structural_attention_layers):
            attn_outputs = []
            for t in range(self.num_time_steps):
                # print('shape of input', input_list[t].size())
                out = layer([input_list[t], adjs[t]])
                out = F.dropout(out, self.spatial_drop)
                # print("shape of struction att out", out.shape)
                attn_outputs.append(out)  # A list of [1x Ni x F]
            input_list = list(attn_outputs)

        # 2: Pack embeddings across snapshots.
        # for t in range(0, self.num_time_steps):
        #     zero_padding = tf.zeros(
        #         [1, tf.shape(attn_outputs[-1])[1] - tf.shape(attn_outputs[t])[1], self.structural_attention_layers[-1]]) # 时间掩码，防止聚合未来时间节点信息
        #     attn_outputs[t] = tf.concat([attn_outputs[t], zero_padding], axis=1)

        structural_outputs = torch.stack(attn_outputs).permute(1, 0, 2)  # [N, T, F]
        # for t in range(0, self.num_time_steps):
        #     attn_outputs[t] = torch.unsqueeze(attn_outputs[t], dim=1)
        # structural_outputs = torch.cat(attn_outputs, dim=1)
        print("structural_outputs: ", structural_outputs.size())

        # # 3: Temporal Attention forward
        temporal_inputs = structural_outputs
        for temporal_layer in self.temporal_attention_layers:
            outputs = temporal_layer(temporal_inputs, )  # [N, T, F]
            outputs = F.dropout(outputs, p=self.temporal_drop)
            temporal_inputs = outputs
            self.attn_wts_all.append(temporal_layer.attn_wts_all)

        self.final_output_embeddings = outputs[:, -1, :]

        # self.final_output_embeddings = structural_outputs[:, -1, :]
        print("shape of final output embedding:", self.final_output_embeddings.size())
        # 4: 添加全连接层
        embed = F.relu(self.embed_layer(self.final_output_embeddings))
        # 添加sigmoid激活函数用于二分类
        self.logist = self.fc_layer(embed)
        self.pred_y = torch.squeeze(torch.round(torch.sigmoid(self.logist)).to(torch.int32))  # 预测标签
        # print(self.logist)
        self.labels = placeholders['label'].to(device)

        return self.logist, self.final_output_embeddings

    def _loss(self):
        # 分类损失，采用sigmoid交叉熵损失
        self.class_loss = 0.0
        logs = F.binary_cross_entropy_with_logits(self.logist.squeeze(), self.labels.float())
        self.class_loss = torch.mean(logs)
        print("class_loss", self.class_loss)
        # 嵌入损失，使同类型节点嵌入尽可能相似，不同类型节点嵌入尽可能远离
        self.graph_loss = 0.0
        pos_idx = torch.squeeze(torch.nonzero(self.labels == 1))
        pos_embeds = self.final_output_embeddings[pos_idx]  # 正样本的嵌入
        pos_norms = torch.norm(pos_embeds, p=2, dim=1, keepdim=True)
        pos_embeds = pos_embeds / pos_norms  # 单位向量
        # print('size of pos_embeds', pos_embeds.size())
        neg_idx = torch.squeeze(torch.nonzero(self.labels == 0))
        neg_embeds = self.final_output_embeddings[neg_idx]  # 负样本嵌入
        neg_norms = torch.norm(neg_embeds, p=2, dim=1, keepdim=True)
        neg_embeds = neg_embeds / neg_norms  # 单位向量
        # print('size of neg_embeds', neg_embeds.size())

        pos_score_1 = torch.mean(torch.matmul(pos_embeds, pos_embeds.permute(1, 0)), dim=1)
        pos_score_2 = torch.mean(torch.matmul(neg_embeds, neg_embeds.permute(1, 0)), dim=1)
        pos_score = torch.cat([pos_score_1, pos_score_2], dim=0) + 1e-5
        # print("shape of pos_score: ", pos_score)
        neg_score_1 = torch.mean((-1.0) * torch.matmul(pos_embeds, neg_embeds.permute(1, 0)), dim=1)
        neg_score_2 = torch.mean((-1.0) * torch.matmul(neg_embeds, pos_embeds.permute(1, 0)), dim=1)
        neg_score = torch.cat([neg_score_1, neg_score_2], dim=0) + 1e-5
        # # print("shape of neg_score: ", neg_score)
        # graph_logs_1 = F.binary_cross_entropy_with_logits(pos_score_1, torch.ones_like(pos_score_1))
        # graph_logs_2 = F.binary_cross_entropy_with_logits(neg_score_1, torch.ones_like(neg_score_1))
        graph_logs_1 = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        graph_logs_2 = F.binary_cross_entropy_with_logits(neg_score, torch.ones_like(neg_score))
        self.graph_loss = graph_logs_1 + graph_logs_2
        print("graph_loss:", self.graph_loss)

        self.reg_loss = 0.0
        for name, v in self.named_parameters():
            # print('all train_variables:', v.name)
            if "attention_layers" in name and "bias" not in name:
                # print('train_variables:', name, tf.reduce_mean(torch.norm(v, p=2) * cfg.weight_decay)
                self.reg_loss = self.reg_loss + torch.mean(torch.norm(v, p=2) * cfg.weight_decay)
        print('reg_loss', self.reg_loss)
        self.loss = cfg.a * self.class_loss + cfg.b * self.graph_loss + cfg.g * self.reg_loss
        return self.loss, self.class_loss, self.graph_loss, self.reg_loss


    def _result_score(self):
        # pred_y = torch.squeeze(torch.round(F.sigmoid(self.logist)))  # 预测标签
        print("pred_y", self.pred_y)
        assert self.pred_y.shape == self.labels.shape
        accuracy = Accuracy()
        accuracy.update(self.pred_y, self.labels)
        accuracy = accuracy.compute().item()  # 获取准确率并转换为 NumPy 数组
        # print('acc', accuracy)
        recall = Recall()
        recall.update(self.pred_y, self.labels)
        recall = recall.compute().item()
        # print('recall', recall)
        precision = Precision()
        precision.update(self.pred_y, self.labels)
        precision = precision.compute().item()
        # print('precision', precision)
        # Auc = AUC()
        # Auc.update(self.pred_y, self.labels)
        # auc = Auc.compute().item()
        fpr, tpr, thresholds = roc_curve(self.labels, self.pred_y, pos_label=1)
        AUC = auc(fpr, tpr)
        # print('AUC', AUC)
        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-9))
        # print('f1_score', f1_score)
        self.result_scores = [accuracy, recall, precision, AUC, f1_score]
        return self.result_scores
