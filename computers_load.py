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


def load_computers(per_classnum, seed, thred=0.9):
    """
    修复label_encoded问题的Computers数据集加载器
    """
    args = set_params()

    # 1. 加载原始数据并添加label_encoded
    df_orig = load_and_preprocess_data('./dataset/computer/Computers.csv', per_classnum)
    le = LabelEncoder()
    df_orig['label_encoded'] = le.fit_transform(df_orig['label'])  # 确保原始数据有编码列

    # 2. 加载改写数据并添加label_encoded
    try:
        df_rewritten = load_and_preprocess_data(f"./dataset/computer/Computers_ratio{args.ratio}.csv",
                                                per_classnum)
        df_rewritten['label_encoded'] = le.transform(df_rewritten['label'])  # 使用相同的编码器
    except FileNotFoundError:
        print(f"警告: 未找到改写数据，仅使用原始数据")
        df_rewritten = pd.DataFrame(columns=df_orig.columns)  # 创建空DataFrame保持结构

    # 3. 合并数据
    df_combined = pd.concat([df_orig, df_rewritten], ignore_index=True)
    print(f"总节点数: {len(df_combined)} (原始:{len(df_orig)} + 改写:{len(df_rewritten)})")

    # 4. 特征工程
    x_orig, y_orig, edge_index_orig = process_features_and_edges(df_orig)
    if not df_rewritten.empty:
        x_rewritten, y_rewritten, _ = process_features_and_edges(df_rewritten)
    else:
        x_rewritten, y_rewritten = torch.empty((0, x_orig.size(1))), torch.empty((0,), dtype=torch.long)

    # 5. 合并特征
    combined_x = torch.cat([x_orig, x_rewritten], dim=0)
    combined_y = torch.cat([y_orig, y_rewritten], dim=0)

    # 6. 处理图结构
    N_orig = len(df_orig)
    edge_index = merge_edges(df_orig, df_rewritten, edge_index_orig, N_orig)

    # 7. 边预测（仅当有改写数据时）
    if not df_rewritten.empty:
        text_embed_orig = x_orig[:, -384:]
        text_embed_rewritten = x_rewritten[:, -384:]
        sim_matrix = cosine_similarity(text_embed_rewritten, text_embed_orig)

        new_edges = []
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                if sim_matrix[i, j] > thred:
                    src = i + N_orig
                    dst = j
                    new_edges.extend([[src, dst], [dst, src]])
        num_new_edges = len(new_edges)
        print(f"\n边预测结果:")
        print(f"  原始边数量: {edge_index_orig.shape[1]}")
        print(f"  新增边数量: {num_new_edges}")
        if new_edges:
            edge_index = torch.cat([edge_index, torch.tensor(new_edges).t().contiguous()], dim=1)

    # 8. 数据集划分（兼容无改写数据的情况）
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

    # 验证/测试集划分
    remaining = [i for i in range(total_nodes) if not train_mask[i]]
    np.random.shuffle(remaining)
    test_val_size = ceil(len(remaining) * 1)
    test_size = ceil(test_val_size * 0.5)

    test_mask[remaining[:test_size]] = True
    val_mask[remaining[test_size:test_val_size]] = True

    data = Data(
        x=combined_x,
        edge_index=edge_index,
        y=combined_y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    return data, combined_x


# 辅助函数保持不变
def load_and_preprocess_data(filepath, per_classnum):
    df = pd.read_csv(filepath, engine='python', on_bad_lines='skip')
    df = df[df['label'].notnull()]
    if 'year' in df.columns and 'len' in df.columns:
        df = df.dropna(subset=['year', 'len'])
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= (per_classnum + 3)].index
    df = df[df['label'].isin(valid_labels)].copy()
    return df.reset_index(drop=True)


def process_features_and_edges(df):
    enc = OneHotEncoder(sparse=False)
    cat_feat = enc.fit_transform(df[['category']])
    year_feat = MinMaxScaler().fit_transform(df[['year']])
    len_feat = MinMaxScaler().fit_transform(df[['len']])
    text_feat = model.encode(df['text'].tolist(), show_progress_bar=True)
    x_np = np.concatenate([cat_feat, year_feat, len_feat, text_feat], axis=1)

    # 这里不再需要label_encoder，因为已在主函数处理
    y = torch.tensor(df['label_encoded'].values, dtype=torch.long)

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

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0),
                                                                                                          dtype=torch.long)
    return torch.tensor(x_np, dtype=torch.float), y, edge_index


def merge_edges(df_orig, df_rewritten, orig_edge_index, N_orig):
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