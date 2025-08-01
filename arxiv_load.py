import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import gc
from math import ceil
from utils import set_params
from torch_geometric.utils import remove_self_loops
from model_utils import get_model
model = get_model()
# 检查是否有可用的 GPU
args = set_params()
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def build_similarity_edges_gpu_batched(text_feat: np.ndarray, thred: float, train_mask: torch.Tensor,
                                       test_mask: torch.Tensor, k: int = 20, batch_size: int = 2048):
    """
    使用 GPU (PyTorch) 分块高效构建相似度边，以节省显存。
    修改：增加 train_mask 和 test_mask 参数，防止生成跨训练/测试集的边

    Args:
        text_feat (np.ndarray): 文本特征的 numpy 数组。
        thred (float): 相似度阈值。
        train_mask (torch.Tensor): 训练集节点掩码。
        test_mask (torch.Tensor): 测试集节点掩码。
        k (int): 每个节点考虑的最大邻居数。
        batch_size (int): 处理节点的批大小，可根据显存调整。

    Returns:
        torch.Tensor: 边索引 (edge_index)。
        float: 计算出的平均相似度，用于可能的阈值调整。
    """
    print(f"在 GPU 上分块构建相似度边 (batch_size={batch_size})...")
    n_nodes = len(text_feat)

    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    feat_tensor = torch.tensor(text_feat, dtype=torch.float32, device=device)
    feat_tensor = torch.nn.functional.normalize(feat_tensor, p=2, dim=1)

    all_source_nodes = []
    all_target_nodes = []
    all_similarities_list = []

    for i in range(0, n_nodes, batch_size):
        batch_start = i
        batch_end = min(i + batch_size, n_nodes)
        batch_feat = feat_tensor[batch_start:batch_end]

        similarity_matrix_batch = batch_feat @ feat_tensor.T

        k_actual = min(k + 1, n_nodes)
        top_k_sims, top_k_indices = torch.topk(similarity_matrix_batch, k_actual, dim=1)

        # 移除自身
        top_k_sims = top_k_sims[:, 1:]
        top_k_indices = top_k_indices[:, 1:]

        # 创建源节点索引
        source_nodes = torch.arange(batch_start, batch_end, device=device).unsqueeze(1).expand(-1, k_actual - 1)

        # 关键修改：确保所有张量都在GPU上
        batch_train_mask = train_mask[batch_start:batch_end].to(device)
        target_train_mask = train_mask[top_k_indices].to(device)
        target_test_mask = test_mask[top_k_indices].to(device)

        mask = (top_k_sims > thred) & (
                (batch_train_mask.unsqueeze(1) & target_train_mask) |  # 训练-训练
                (~batch_train_mask.unsqueeze(1) & ~target_train_mask & ~target_test_mask) |  # 验证-验证
                (~batch_train_mask.unsqueeze(1) & target_test_mask)  # 测试-测试
        )

        all_source_nodes.append(source_nodes[mask])
        all_target_nodes.append(top_k_indices[mask])
        all_similarities_list.append(top_k_sims[mask].cpu())

    # 清理显存
    del feat_tensor, similarity_matrix_batch, top_k_sims, top_k_indices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 汇总所有批次的结果
    if not all_source_nodes:
        print("警告: 在当前阈值下没有生成任何边。")
        return torch.tensor([], dtype=torch.long), 0.0

    source_nodes = torch.cat(all_source_nodes)
    target_nodes = torch.cat(all_target_nodes)

    # 计算平均相似度
    all_similarities = torch.cat(all_similarities_list)
    mean_sim = all_similarities.mean().item()
    print(f"已发现边的相似度统计: min={all_similarities.min():.3f}, max={all_similarities.max():.3f}, "
          f"mean={mean_sim:.3f}, thred={thred}")

    # 构建无向图的 edge_index
    edge_list = torch.stack([source_nodes, target_nodes])
    undirected_edge_index = torch.cat([edge_list, edge_list.flip(0)], dim=1)
    final_edge_index = torch.unique(undirected_edge_index, dim=1).cpu()
    num_edges = final_edge_index.shape[1]
    print(f"生成边统计: 共添加 {num_edges} 条边 (阈值={thred}, 平均相似度={mean_sim:.3f})")
    return final_edge_index, mean_sim
def load_arxiv(per_classnum, seed, thred, batch_size=500):
    # 加载原始数据和改写数据
    args = set_params()
    df_original = pd.read_csv('./dataset/Arxiv/Arxiv.csv', engine='python', on_bad_lines='skip')
    df_rewrite = pd.read_csv(f'./dataset/Arxiv/Arxiv_ratio{args.ratio}.csv', engine='python', on_bad_lines='skip')

    # 确保改写数据有对应的原始数据ID
    df_rewrite = df_rewrite[df_rewrite['ID'].notnull()]

    # 数据预处理
    df_original = df_original[df_original['label_id'].notnull()]
    df_original = df_original.dropna(subset=['title', 'abstract'])
    df_rewrite = df_rewrite[df_rewrite['label_id'].notnull()]
    df_rewrite = df_rewrite.dropna(subset=['title', 'abstract'])

    # 过滤样本量不足的类别（考虑原始数据和改写数据）
    original_label_counts = df_original['label_id'].value_counts()
    rewrite_label_counts = df_rewrite['label_id'].value_counts()

    # 只保留在两个数据集中都足够的类别
    valid_labels = set(original_label_counts[original_label_counts >= (per_classnum + 3)].index) & \
                   set(rewrite_label_counts[rewrite_label_counts >= per_classnum].index)

    df_original = df_original[df_original['label_id'].isin(valid_labels)].copy()
    df_rewrite = df_rewrite[df_rewrite['label_id'].isin(valid_labels)].copy()

    # 为改写数据添加标记
    df_rewrite['is_rewrite'] = True
    df_original['is_rewrite'] = False

    # 合并数据集
    df_combined = pd.concat([df_original, df_rewrite], ignore_index=True)
    df_combined = df_combined.reset_index(drop=True)

    print(
        f"过滤后剩余类别数: {len(valid_labels)}, 原始数据样本数: {len(df_original)}, 改写数据样本数: {len(df_rewrite)}")

    # 类别特征
    enc = OneHotEncoder(sparse=False)
    cat_feat = enc.fit_transform(df_combined[['category']])

    # 分批处理文本嵌入（原始和改写）
    combined_text = (df_combined['title'] + " " + df_combined['abstract']).tolist()
    text_feat = []
    for i in range(0, len(combined_text), batch_size):
        batch = combined_text[i:i + batch_size]
        text_feat.append(model.encode(batch, show_progress_bar=False))
        gc.collect()

    text_feat = np.concatenate(text_feat, axis=0)

    # 合并特征
    x_np = np.concatenate([cat_feat, text_feat], axis=1)
    x = torch.tensor(x_np, dtype=torch.float)
    y = torch.tensor(df_combined['label_id'].values, dtype=torch.long)

    np.random.seed(seed)
    train_mask = torch.zeros(len(df_combined), dtype=torch.bool)
    val_mask = torch.zeros(len(df_combined), dtype=torch.bool)
    test_mask = torch.zeros(len(df_combined), dtype=torch.bool)

    for label_id in valid_labels:
        # 获取原始数据和改写数据的索引
        original_indices = df_combined[(df_combined['label_id'] == label_id) &
                                       (~df_combined['is_rewrite'])].index.tolist()
        rewrite_indices = df_combined[(df_combined['label_id'] == label_id) &
                                      (df_combined['is_rewrite'])].index.tolist()

        if len(original_indices) < per_classnum + 3 or len(rewrite_indices) < per_classnum:
            continue

        # 随机选择perclassnum个改写数据和对应的原始数据作为训练集
        np.random.shuffle(rewrite_indices)
        selected_rewrites = rewrite_indices[:per_classnum]

        # 获取这些改写数据对应的原始数据
        selected_originals = []
        for rewrite_idx in selected_rewrites:
            original_id = df_combined.loc[rewrite_idx, 'ID']
            original_idx = df_combined[(df_combined['ID'] == original_id) &
                                       (df_combined['label_id'] == label_id)].index[0]
            selected_originals.append(original_idx)

        # 合并训练样本
        train_samples = selected_rewrites + selected_originals

        # 剩余的原始数据用于验证和测试
        remaining_originals = [idx for idx in original_indices if idx not in selected_originals]
        np.random.shuffle(remaining_originals)
        val_test_split = ceil(len(remaining_originals) * 0.5)

        train_mask[train_samples] = True
        val_mask[remaining_originals[:val_test_split]] = True
        test_mask[remaining_originals[val_test_split:]] = True

    edge_index, mean_sim = build_similarity_edges_gpu_batched(
        text_feat, thred, train_mask, test_mask, batch_size=2048
    )
    edge_index, _ = remove_self_loops(edge_index)

    # 创建Data对象
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # 添加改写标记作为额外属性
    data.is_rewrite = torch.tensor(df_combined['is_rewrite'].values, dtype=torch.bool)
    data.original_id = torch.tensor(df_combined['ID'].fillna(-1).values, dtype=torch.long)

    # 验证是否有数据泄露
    test_train_edges = data.edge_index[:, (data.train_mask[data.edge_index[0]] & data.test_mask[data.edge_index[1]])]
    print(f"测试集与训练集的连接边数: {test_train_edges.shape[1]}")
    if test_train_edges.shape[1] > 0:
        print("警告: 存在数据泄露！测试集与训练集通过边直接连接。")
    else:
        print("验证通过：训练集和测试集之间没有边连接")

    return data, data.x