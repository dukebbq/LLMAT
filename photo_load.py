import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from math import ceil
from sklearn.metrics.pairwise import cosine_similarity
from utils import set_params
from model_utils import get_model
model = get_model()


def load_photo(per_classnum, seed, thred=1.0):
    """
    加载Photo数据集，保持与history_load相同的接口格式

    参数:
        per_classnum: 每类训练样本数
        seed: 随机种子
        thred: 边预测相似度阈值(默认0.1)

    返回:
        data: 图数据对象
        combined_x: 合并后的节点特征
    """
    args = set_params()

    # 1. 加载原始数据
    df_orig = load_and_preprocess_data('./dataset/photo/Photo.csv', per_classnum)

    # 2. 加载改写数据
    df_rewritten = load_and_preprocess_data(
        f"./dataset/photo/Photo_ratio{args.ratio}.csv",
        per_classnum
    )

    # 3. 合并数据
    df_combined = pd.concat([df_orig, df_rewritten], ignore_index=True)
    df_combined = df_combined.reset_index(drop=True)

    # 4. 特征工程
    x_orig, y_orig, edge_index_orig = process_features_and_edges(df_orig)
    x_rewritten, y_rewritten, _ = process_features_and_edges(df_rewritten)

    # 5. 合并特征
    combined_x = torch.cat([x_orig, x_rewritten], dim=0)
    combined_y = torch.cat([y_orig, y_rewritten], dim=0)

    # 6. 处理图结构
    N_orig = len(df_orig)
    N_rewritten = len(df_rewritten)

    # 调整改写数据的节点ID
    rewritten_edge_list = []
    node_id_map = {nid: idx + N_orig for idx, nid in enumerate(df_rewritten['node_id'])}
    for _, row in df_rewritten.iterrows():
        try:
            src = node_id_map[row['node_id']]
            for dst_raw in str(row['neighbour']).split(','):
                dst_raw = dst_raw.strip().lstrip('[').rstrip(']')
                dst_id = int(dst_raw)
                if dst_id in node_id_map:
                    dst = node_id_map[dst_id]
                    rewritten_edge_list.append([src, dst])
        except:
            continue

    # 合并边
    if len(rewritten_edge_list) > 0:
        rewritten_edge_index = torch.tensor(rewritten_edge_list, dtype=torch.long).t().contiguous()
        combined_edge_index = torch.cat([edge_index_orig, rewritten_edge_index], dim=1)
    else:
        combined_edge_index = edge_index_orig

    # 7. 边预测 - 基于文本相似度添加新边
    # 计算文本嵌入相似度
    text_embed_orig = x_orig[:, -384:]  # 假设最后384维是文本嵌入
    text_embed_rewritten = x_rewritten[:, -384:]
    sim_matrix = cosine_similarity(text_embed_rewritten.numpy(), text_embed_orig.numpy())

    # 找到相似度大于阈值的节点对
    new_edges = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if sim_matrix[i, j] > thred:
                # 改写数据节点ID需要偏移
                src = i + N_orig
                dst = j
                new_edges.append([src, dst])
                new_edges.append([dst, src])  # 无向图添加双向边
    num_new_edges = len(new_edges)
    print(f"\n边预测结果:")
    print(f"  原始边数量: {edge_index_orig.shape[1]}")
    print(f"  改写边数量: {len(rewritten_edge_list) if len(rewritten_edge_list) > 0 else 0}")
    print(f"  新增边数量: {num_new_edges}")
    print(
        f"  总边数量: {combined_edge_index.shape[1] + num_new_edges if num_new_edges > 0 else combined_edge_index.shape[1]}")
    if len(new_edges) > 0:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        combined_edge_index = torch.cat([combined_edge_index, new_edge_index], dim=1)

    # 8. 数据集划分
    np.random.seed(seed)
    total_nodes = N_orig + N_rewritten
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)

    # 获取改写节点对应的原始节点ID
    rewritten_ids = df_rewritten['node_id'].values
    corresponding_orig_indices = df_orig[df_orig['node_id'].isin(rewritten_ids)].index.tolist()

    # 训练集：改写数据的per_classnum节点 + 对应的原始节点
    for label_id in df_rewritten['label_encoded'].unique():
        # 从改写数据中选择per_classnum个节点
        rewritten_indices = df_rewritten[df_rewritten['label_encoded'] == label_id].sample(
            n=min(per_classnum, len(df_rewritten[df_rewritten['label_encoded'] == label_id])),
            random_state=seed
        ).index.tolist()

        # 找到对应的原始节点
        selected_rewritten_ids = df_rewritten.loc[rewritten_indices, 'node_id']
        orig_indices = df_orig[df_orig['node_id'].isin(selected_rewritten_ids)].index.tolist()

        # 设置掩码（改写节点需要偏移N_orig）
        train_mask[[i + N_orig for i in rewritten_indices]] = True
        train_mask[orig_indices] = True

    # 验证/测试集：剩余所有节点
    remaining_indices = [i for i in range(total_nodes) if not train_mask[i]]
    np.random.shuffle(remaining_indices)
    n_test_val = ceil(len(remaining_indices) * 1.0)  # 30%用于测试+验证
    n_test = ceil(n_test_val * 0.5)  # 15%测试
    n_val = n_test_val - n_test  # 15%验证

    test_mask[remaining_indices[:n_test]] = True
    val_mask[remaining_indices[n_test:n_test + n_val]] = True

    data = Data(
        x=combined_x,
        edge_index=combined_edge_index,
        y=combined_y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data, combined_x


def load_and_preprocess_data(filepath, per_classnum):
    """加载和预处理Photo数据集"""
    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip')
    df = df[df['label'].notnull()]

    # 处理特殊字段
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['len'] = pd.to_numeric(df['len'], errors='coerce')

    # 过滤样本量不足的类别
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= (per_classnum + 3)].index
    df = df[df['label'].isin(valid_labels)].copy()
    df = df.dropna(subset=['year', 'len'])

    return df.reset_index(drop=True)


def process_features_and_edges(df):
    """处理Photo数据集的特征和边关系"""
    # 1. 类别特征
    enc = OneHotEncoder(sparse=False)
    cat_feat = enc.fit_transform(df[['category']])

    # 2. 数值特征
    num_feat = MinMaxScaler().fit_transform(df[['year', 'len']])

    # 3. 文本特征（合并text和short_text）
    full_texts = df['text'] + " " + df['short_text']
    text_feat = model.encode(full_texts.tolist(), show_progress_bar=True)

    # 4. 合并所有特征
    x_np = np.concatenate([cat_feat, num_feat, text_feat], axis=1)
    x = torch.tensor(x_np, dtype=torch.float)

    # 标签编码
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    y = torch.tensor(df['label_encoded'].values, dtype=torch.long)

    # 构建边
    edge_list = []
    node_id_map = {nid: idx for idx, nid in enumerate(df['node_id'])}
    for _, row in df.iterrows():
        try:
            src = node_id_map[row['node_id']]
            for dst_raw in str(row['neighbour']).split(','):
                dst_raw = dst_raw.strip().lstrip('[').rstrip(']')
                dst_id = int(dst_raw)
                if dst_id in node_id_map:
                    dst = node_id_map[dst_id]
                    edge_list.append([src, dst])
        except:
            continue

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0),
                                                                                                          dtype=torch.long)

    return x, y, edge_index