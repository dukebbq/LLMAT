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


def load_children(per_classnum, seed, thred=0.1):
    """
    基于history模板修改的Children数据集加载器
    字段: category, text, label, node_id, neighbour
    """
    args = set_params()

    # 1. 加载原始数据
    df_orig = load_and_preprocess_data('./dataset/Children/Children.csv', per_classnum)

    # 2. 加载改写数据（需确保文件存在）
    try:
        df_rewritten = load_and_preprocess_data(f"./dataset/Children/Children_ratio{args.ratio}.csv",
                                                per_classnum)
    except FileNotFoundError:
        print(f"警告: 未找到改写数据，仅使用原始数据")
        df_rewritten = pd.DataFrame(columns=df_orig.columns)

    # 3. 标签编码（统一处理原始和改写数据）
    le = LabelEncoder()
    df_orig['label_encoded'] = le.fit_transform(df_orig['label'])
    if not df_rewritten.empty:
        df_rewritten['label_encoded'] = le.transform(df_rewritten['label'])

    # 4. 合并数据
    df_combined = pd.concat([df_orig, df_rewritten], ignore_index=True)
    print(f"总节点数: {len(df_combined)} (原始:{len(df_orig)} + 改写:{len(df_rewritten)})")

    # 5. 特征工程
    x_orig, y_orig, edge_index_orig = process_features_and_edges(df_orig)
    if not df_rewritten.empty:
        x_rewritten, y_rewritten, _ = process_features_and_edges(df_rewritten)
    else:
        x_rewritten, y_rewritten = torch.empty((0, x_orig.size(1))), torch.empty((0,), dtype=torch.long)

    # 6. 合并特征
    combined_x = torch.cat([x_orig, x_rewritten], dim=0)
    combined_y = torch.cat([y_orig, y_rewritten], dim=0)

    # 7. 处理图结构
    N_orig = len(df_orig)
    edge_index = merge_edges(df_orig, df_rewritten, edge_index_orig, N_orig)

    # 8. 边预测（仅当有改写数据时）
    if not df_rewritten.empty:
        text_embed_orig = x_orig[:, -384:]  # 文本特征维度
        text_embed_rewritten = x_rewritten[:, -384:]
        sim_matrix = cosine_similarity(text_embed_rewritten, text_embed_orig)

        new_edges = []
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                if sim_matrix[i, j] > thred:
                    src = i + N_orig  # 改写节点ID偏移
                    dst = j
                    new_edges.extend([[src, dst], [dst, src]])  # 无向图
        num_new_edges = len(new_edges)
        print(f"\n边预测结果:")
        print(f"  原始边数量: {edge_index_orig.shape[1]}")
        print(f"  新增边数量: {num_new_edges}")
        if new_edges:
            edge_index = torch.cat([edge_index, torch.tensor(new_edges).t().contiguous()], dim=1)

    # 9. 数据集划分（兼容无改写数据情况）
    np.random.seed(seed)
    total_nodes = len(df_combined)
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)

    if not df_rewritten.empty:
        # 有改写数据时的划分逻辑
        for label_id in df_rewritten['label_encoded'].unique():
            rewritten_indices = df_rewritten[df_rewritten['label_encoded'] == label_id].sample(
                n=min(per_classnum, sum(df_rewritten['label_encoded'] == label_id)),
                random_state=seed
            ).index.tolist()

            # 找到对应原始节点
            corresponding_ids = df_rewritten.loc[rewritten_indices, 'node_id']
            orig_indices = df_orig[df_orig['node_id'].isin(corresponding_ids)].index.tolist()

            train_mask[[i + N_orig for i in rewritten_indices]] = True
            train_mask[orig_indices] = True
    else:
        # 无改写数据时的原始划分逻辑
        for label_id in df_orig['label_encoded'].unique():
            indices = df_orig[df_orig['label_encoded'] == label_id].index.tolist()
            if len(indices) < per_classnum + 3:
                continue
            np.random.shuffle(indices)
            train_mask[indices[:per_classnum]] = True

    # 验证/测试集划分（剩余节点的30%，其中测试和验证各半）
    remaining = [i for i in range(total_nodes) if not train_mask[i]]
    np.random.shuffle(remaining)
    test_val_size = ceil(len(remaining) * 0.3)  # 30%用于测试+验证
    test_size = ceil(test_val_size * 0.5)  # 15%测试
    val_size = test_val_size - test_size  # 15%验证

    test_mask[remaining[:test_size]] = True
    val_mask[remaining[test_size:test_size + val_size]] = True

    data = Data(
        x=combined_x,
        edge_index=edge_index,
        y=combined_y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    return data, combined_x


# ---------------------- 辅助函数 ----------------------
def load_and_preprocess_data(filepath, per_classnum):
    """数据加载与预处理（适配Children数据集）"""
    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip')
    df = df[df['label'].notnull()]
    if per_classnum > 9:
        label_counts = df['label'].value_counts()
        valid_labels = label_counts[label_counts >= 100].index
        df = df[df['label'].isin(valid_labels)].copy()
    # 过滤样本量不足的类别
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= (per_classnum + 3)].index
    df = df[df['label'].isin(valid_labels)].copy()

    # 检查必要字段
    required_columns = ['category', 'node_id', 'neighbour', 'text']
    assert all(col in df.columns for col in required_columns), f"缺少必要字段: {required_columns}"

    return df.reset_index(drop=True)


def process_features_and_edges(df):
    """特征和边处理（适配Children字段）"""
    # 特征工程
    enc = OneHotEncoder(sparse=False)
    cat_feat = enc.fit_transform(df[['category']])
    text_feat = model.encode(df['text'].tolist(), show_progress_bar=True)
    x_np = np.concatenate([cat_feat, text_feat], axis=1)

    # 标签已在主函数处理
    y = torch.tensor(df['label_encoded'].values, dtype=torch.long)

    # 构建边
    edge_list = []
    node_id_map = {nid: idx for idx, nid in enumerate(df['node_id'])}
    for _, row in df.iterrows():
        try:
            src = node_id_map[row['node_id']]
            for dst in str(row['neighbour']).strip("[]").split(','):
                dst = dst.strip()
                if dst:
                    dst = int(dst)
                    if dst in node_id_map:
                        edge_list.append([src, node_id_map[dst]])
        except:
            continue

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return torch.tensor(x_np, dtype=torch.float), y, edge_index


def merge_edges(df_orig, df_rewritten, orig_edge_index, N_orig):
    """边合并逻辑（与history模板一致）"""
    edge_list = orig_edge_index.t().tolist()
    if not df_rewritten.empty:
        node_id_map = {nid: idx + N_orig for idx, nid in enumerate(df_rewritten['node_id'])}
        for _, row in df_rewritten.iterrows():
            try:
                src = node_id_map[row['node_id']]
                for dst in str(row['neighbour']).strip("[]").split(','):
                    dst = dst.strip()
                    if dst:
                        dst = int(dst)
                        if dst in node_id_map:
                            edge_list.append([src, node_id_map[dst]])
            except:
                continue
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()