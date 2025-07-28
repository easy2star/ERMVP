import torch
import math

def merge_tokens(x, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """
    B, N, C = x.shape
    #B,N
    idx_token = torch.arange(N)[None, :].repeat(B, 1)
    # idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
    agg_weight = x.new_ones(B, N, 1)
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)
    #[[0]]
    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # average token features
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    return x_merged,idx_token_new

def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

# 计算局部密度:
# 对于每个特征向量，计算其与 k 个最近邻的距离。
# 局部密度可以通过这些距离的倒数或其他相似性度量来计算。
# 计算相对距离:
# 对于每个特征向量，计算其到密度更高的最近邻的距离。
# 确定聚类中心:
# 根据局部密度和相对距离，选择聚类中心。
# 通常选择局部密度高且相对距离远的点作为聚类中心。
# 分配聚类:
# 将每个特征向量分配到最近的聚类中心所代表的簇中。

# cluster_dpc_knn 函数通过 DPC-KNN 算法对输入的特征向量进行聚类，返回每个特征向量所属的聚类索引以及实际的聚类数量。
# 这种算法结合了密度信息和最近邻信息，能够有效地处理不同密度的聚类问题。

# src = N,max_num,C   cluster_num = 1408
# cluster_dpc_knn 函数实现了一种基于密度峰值聚类（Density Peak Clustering, DPC）结合 k-最近邻（k-Nearest Neighbors, kNN）的聚类算法，用于对输入的特征向量 x 进行聚类。
def cluster_dpc_knn(x, cluster_num, k=5):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
            
        # idx_cluster: 一个形状为 [B, N] 的张量，表示每个特征向量所属的聚类索引。
        
        # cluster_num: 实际的聚类数量，通常与输入的 cluster_num 相同。
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
            
        # x: 输入的特征向量，形状为 [B, N, D]，其中：
        # B 是批量大小（batch size）。
        # N 是每个样本中的特征数量（例如，图像中的像素点、点云中的点等）。
        # D 是每个特征的维度。
        
        # cluster_num: 目标聚类数量，即希望将输入特征划分为多少个簇。
        
        # k: 用于计算局部密度时考虑的最近邻数量，默认值为 5。
    """
    with torch.no_grad():
        B, N, C = x.shape    # N,max_num,C
        # print(x.shape)
        # exit()
        # 批量计算两个向量集合的距离(默认为欧氏距离)
        # B,N,N
        dist_matrix = torch.cdist(x, x) / (C ** 0.5) 

        # 计算输入张量 x 中所有特征向量之间的欧几里得距离矩阵，并对距离进行了归一化处理，归一化因子为特征向量维度 C 的平方根。最终输出的形状为 [N, max_num, max_num]。
        # 这种操作通常用于聚类算法或其他需要计算距离矩阵的任务中，归一化可以避免因特征维度不同而导致的距离差异。
        # 1. torch.cdist(x, x)
        # torch.cdist 的作用：计算两个张量中所有向量对之间的欧几里得距离。
        # 语法：torch.cdist(x1, x2)，其中 x1 和 x2 是形状为 [m, d] 和 [n, d] 的张量，输出是一个形状为 [m, n] 的距离矩阵。
        # 具体操作：
        # 在这里，x 是形状为 [N, max_num, C] 的张量。
        # torch.cdist(x, x) 会计算 x 中每个特征向量之间的欧几里得距离。
        # 由于 x 是三维张量，torch.cdist 会自动处理批量维度（N），因此输出的形状是 [N, max_num, max_num]。
        # 2. / (C ** 0.5)
        # 归一化的作用：将距离矩阵中的每个元素除以 C 的平方根（C ** 0.5）。
        # 目的：这种归一化可以减少特征向量维度 C 对距离的影响。当 C 较大时，欧几里得距离可能会因为维度的增加而变得较大，通过除以 C ** 0.5，可以将距离标准化到一个合理的范围。
        # 输出形状
        # dist_matrix 的形状是 [N, max_num, max_num]，表示每个车辆的 max_num 个特征向量之间的距离矩阵。

        # print(dist_matrix.shape)
        # get local density
        # B,N,K
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        # 从形状为 [N, max_num, max_num] 的距离矩阵 dist_matrix 中，为每个特征向量找到其最近的 k 个特征向量的距离和索引。
        # 具体来说，k=10 表示找到每个特征向量的 10 个最近邻。
        
        # 从距离矩阵 dist_matrix 中，为每个特征向量找到其最近的 k 个特征向量的距离和索引。
        # 返回的 dist_nearest 和 index_nearest 的形状均为 [N, max_num, k]。
        # dist_nearest：形状为 [N, max_num, k]，表示每个特征向量的 k 个最近邻的距离。
        # index_nearest：形状为 [N, max_num, k]，表示每个特征向量的 k 个最近邻的索引。
        
        # B,N
        # 距离越小越好，所以有负号
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()  # 负值的指数 属于 （0,1） 的范围
        # 基于每个特征向量的 k 个最近邻距离，计算其局部密度。局部密度越高，表示该特征向量周围的特征向量越密集。
        # 这种局部密度计算方法常用于聚类算法（如 DPC-KNN）中，用于评估每个特征向量的局部密度，以便后续选择聚类中心。
        # [N, max_num]，表示每个特征向量的局部密度。    
        
        # print(density.shape)
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
        # get distance indicator
        # [B,N,N]
        mask = density[:, None, :] > density[:, :, None]  # [N, 1, max_num] > [N, max_num, 1]
        # 这行代码的作用是通过比较每个特征向量的局部密度，生成一个掩码（mask）张量，用于指示哪些特征向量的密度更高。
        # 具体来说，它通过广播机制比较每个特征向量的密度值，并生成一个布尔张量。
        # mask: [N, max_num, max_num]
        
        # density[:, None, :] 的形状是 [N, 1, max_num]。
        # density[:, :, None] 的形状是 [N, max_num, 1]。
        # 通过广播机制，这两个张量在比较时会扩展为 [N, max_num, max_num]。
        # 比较结果是一个布尔张量，表示每个特征向量的密度是否大于其他特征向量的密度。

        # print(mask.shape)
        # exit()
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        # [N, max_num, max_num] -> [N, max_num*max_num] -> [N] -> [N,1,1]
        # 从形状为 [N, max_num, max_num] 的距离矩阵 dist_matrix 中计算每个样本的最大距离，并将结果扩展为形状 [N, 1, 1] 的张量。
        # 从每个样本的距离矩阵中找到最大距离，并将结果扩展为形状 [N, 1, 1] 的张量。
        # 这种操作通常用于归一化或其他需要最大距离值的任务中。

        # 这行代码的作用是结合掩码（mask）和最大距离（dist_max）来计算每个特征向量到其“父”特征向量的最小距离，并找到对应的“父”特征向量的索引。
        # 这里的“父”特征向量是指在掩码中被标记为密度更高的特征向量。具体步骤如下：
        # 输入
        # dist_matrix 的形状是 [N, max_num, max_num]，表示每个特征向量之间的距离矩阵。
        # mask 的形状是 [N, max_num, max_num]，布尔张量，表示哪些特征向量的密度更高。
        # dist_max 的形状是 [N, 1, 1]，表示每个样本的最大距离。
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
        # dist：最小值，形状为 [N, max_num]。
        # index_parent：最小值的索引，形状为 [N, max_num]。
        
        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp   = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num

if __name__ == '__main__':
    sample_ratio = 1/2
    src=torch.randn(1,100*256,64)
    cluster_num = max(math.ceil(100*256 * sample_ratio), 1)
    idx_cluster, cluster_num = cluster_dpc_knn(src, cluster_num, 5)
    print(idx_cluster.shape)
    # print(cluster_num.shape)
    down_dict,idx = merge_tokens(src, idx_cluster, cluster_num, None)
    print(down_dict.shape)
    print(idx.shape)
    
