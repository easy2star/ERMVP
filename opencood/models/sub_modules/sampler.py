import torch
from torch import nn
import random
import matplotlib.pyplot as plt
import numpy as np

def show_heatmaps(matrices,path=None, figsize=(5, 5),
                  cmap='Reds'):
    """显示矩阵热图"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)

    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
                pcm = ax.imshow(matrix, cmap=cmap)
    # fig.colorbar(pcm, ax=axes, shrink=0.6)
    # fig.canvas.set_window_title(titles)
    plt.savefig(path,dpi=300)
    # plt.show()

class SortSampler(nn.Module):

    def __init__(self, topk_ratio, input_dim, score_pred_net='2layer-fc-256'):
        super().__init__()
        self.topk_ratio = topk_ratio
        # print(self.topk_ratio)
        # self.topk_ratio = random.uniform(0.05,0.25)
        if score_pred_net == '2layer-fc-256':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, input_dim, 1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(input_dim, 1, 1))
        elif score_pred_net == '2layer-fc-32':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, 32, 1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(32, 1, 1))
        elif score_pred_net == '1layer-fc':
            self.score_pred_net = nn.Conv2d(input_dim, 1, 1)
        else:
            raise ValueError

        self.norm_feature = nn.LayerNorm(input_dim,elementwise_affine=False)
        self.v_proj = nn.Linear(input_dim, input_dim)

    def forward(self, src, pos_embed, sample_ratio, dis_priority):

        bs,c ,h, w  = src.shape
        #各位置的分数
        src_dis = dis_priority*src.permute(1,0,2,3)
        src_dis = src_dis.permute(1,0,2,3).float()   #  N,C,H,W
        # print(src_dis.shape)
        sample_weight = self.score_pred_net(src_dis).sigmoid().view(bs,-1)  # N,H*W
        # sample_weight[mask] = sample_weight[mask].clone() * 0.
        # sample_weight.data[mask] = 0.
        sample_weight_clone = sample_weight.clone().detach()

        if sample_ratio==None:
            sample_ratio = self.topk_ratio
        ##max sample number:
        sample_lens = torch.tensor(h * w * sample_ratio).repeat(bs,1).int()  # bs,1  每个元素都是h*w*sample_ratio
        max_sample_num = sample_lens.max()
        
        min_sample_num = sample_lens.min()  # max_sample 与 min_sample是相同的
        sort_order = sample_weight_clone.sort(descending=True,dim=1)[1]    # sort_order的维度是 N，H*W
        # 对N，H*W按照第二维进行降序排列
        # 对每个样本的权重值进行降序排序，并返回排序后的原始位置索引，常用于后续按权重选择 top-k 元素或采样。
        # tensor([[0.1, 0.5, 0.3],
        #         [0.7, 0.2, 0.9]])
        # tensor([[1, 2, 0],
        #         [2, 0, 1]])
      
        sort_confidence_topk = sort_order[:,:max_sample_num]  # 选择根据topk采样的索引
        sort_confidence_topk_remaining = sort_order[:,min_sample_num:]  # 剩余的索引
        ## flatten for gathering
        src = src.flatten(2).permute(2, 0, 1)  # N,C,H,W -> H*W,N,C
        # src.flatten(2)  # 作用：从第 2 维开始（即 H 和 W）展平，变成 [N, C, H*W]
        # .permute(2, 0, 1) # 作用：重新排列维度顺序，把展平后的 H*W 放在最前面，N 放中间，C 放最后
        # 这是 Transformer 中常见的预处理步骤，把图像的每个空间位置（H*W）看作一个 token，每个 token 有 C 个通道特征
        
        src = self.norm_feature(src)
        # nn.LayerNorm 默认是对最后一维做归一化，因此：
        # 输入必须是 [*, input_dim] 的形状。
        # 你现在的 src 是 [H*W, N, C]，最后一维是 C，正好符合要求。
        # elementwise_affine=False 表示不学习缩放和平移参数（即只做归一化，不做 γ 和 β）。


      

        sample_reg_loss = sample_weight.gather(1,sort_confidence_topk).mean()  # N,H*W  sample_weight是各个车辆每个位置的置信度， sort_confidence_topk根据索引选择相应的置信度  N,max_num
        # gather 用于从一个张量中根据索引提取特定的值。语法：tensor.gather(dim, index)，其中 dim 指定在哪一维上进行索引，index 是索引张量。
        # sample_weight.gather(1, sort_confidence_topk) 的结果是一个形状为 [N, max_num] 的张量，其中每个元素是从 sample_weight 中根据 sort_confidence_topk 提取的权重值。
        # 计算提取出来的 [N, max_num] 张量的均值。

        # src是H*W,N,C     sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c)是max_num,N,C    表示根据置信度前k个最大值的索引取出来对应的特征值 -> max_num,N,C
        # sample_weight是N,H*W，是各个车辆每个位置的置信度， sample_weight.gather(1,sort_confidence_topk).permute(1,0).unsqueeze(-1)是N,max_num->max_num,N->max_num,N,1   ， 是根据前k个最大值的索引取出来的置信度
        # 对选取出来的特征向量乘以该位置的置信权重
        src_sampled = src.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c)) * sample_weight.gather(1,sort_confidence_topk).permute(1,0).unsqueeze(-1)
      
        # pos_embed_sampled = pos_embed.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c))

        # pos_embed 是 H*W,N,1  # 0,1，...,HW-1
        # sort_confidence_topk.permute(1, 0)[..., None]是max_num,N,1     #  N,max_num -> max_num,N,1
        # sort_confidence_topk是前k个最大的置信度的索引值，选出相应的位置嵌入
        # 从 pos_embed 中根据 sort_confidence_topk 提取每个样本的 max_num 个位置嵌入值，结果的形状为 [max_num, N, 1]。
        pos_embed_sampled = pos_embed.gather(0, sort_confidence_topk.permute(1, 0)[..., None])

        # max_num,N,C：乘以特征置信度后的特征向量   1维tensor：N个车辆的特征向量的平均值   N,max_num：前k个最大值置信度对应的索引值   max_num, N, 1：N个车辆选取的前max_num个特征向量的位置嵌入
        return src_sampled, sample_reg_loss, sort_confidence_topk, pos_embed_sampled
