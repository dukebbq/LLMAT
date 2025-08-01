import pandas as pd
import numpy as np
import random
import json
import ast
from tqdm import tqdm
from DB import call_doubao_paraphrase  # 调用豆包API
import html
from utils import set_params

def preserve_original_structure(text):
    """保留原始文本中的双引号结构，确保改写后格式一致"""
    if not text:
        return text

    import re
    quoted_pairs = re.findall(r'""(.*?)""', text)
    for i, pair in enumerate(quoted_pairs):
        text = text.replace(f'""{pair}""', f'__QUOTE_{i}__')

    rewritten = call_doubao_paraphrase(text) or text

    for i, pair in enumerate(quoted_pairs):
        rewritten = rewritten.replace(f'__QUOTE_{i}__', f'""{pair}""')

    return rewritten


def rewrite_with_config(
    input_path,
    output_path,
    config,
    ratio=5,
    seed=42
):
    """
    使用字段配置改写指定数据集中的文本字段

    参数：
    - input_path: str，原始CSV路径
    - output_path: str，改写后输出路径
    - config: dict，字段映射字典，应包含以下键：
        - "text_field"
        - "label_field"
        - "node_id_field"
        - "neighbour_field"（可选）
    - ratio: int，每类改写样本数量
    - seed: int，随机种子
    """
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(input_path)

    text_col = config["text_field"]
    label_col = config["label_field"]
    node_id_col = config.get("node_id_field", None)
    neighbour_col = config.get("neighbour_field", None)

    # 分组选样本
    class_counts = df[label_col].value_counts().to_dict()
    to_rewrite_indices = []
    for label, count in class_counts.items():
        if count >= ratio:
            selected = df[df[label_col] == label].sample(n=ratio, random_state=seed).index.tolist()
            to_rewrite_indices.extend(selected)
        else:
            print(f"跳过类别 '{label}' (样本数 {count} < {ratio})")

    # 改写文本字段
    for idx in tqdm(to_rewrite_indices, desc="Rewriting text"):
        original_text = html.unescape(df.at[idx, text_col])
        df.at[idx, text_col] = preserve_original_structure(original_text)

    # 验证邻居字段（可选）
    if neighbour_col and node_id_col:
        def validate_neighbours(row):
            try:
                parsed = ast.literal_eval(row[neighbour_col])
                return isinstance(parsed, list)
            except:
                return False
        df["__neigh_valid__"] = df.apply(validate_neighbours, axis=1)
        print(f"邻居字段合法行数: {df['__neigh_valid__'].sum()} / {len(df)}")
        df.drop(columns=["__neigh_valid__"], inplace=True)

    # 新代码（仅输出被改写的节点）：
    df_rewritten_only = df.loc[to_rewrite_indices]  # 筛选被改写的行
    df_rewritten_only.to_csv(output_path, index=False, quoting=1)
    print(f"\n✅ 改写完成。输出保存于: {output_path}")
    print(f"共改写样本数: {len(to_rewrite_indices)} / {len(df)}")


if __name__ == "__main__":
    # 示例配置，可以存为 config.json 或通过代码传入
    sample_config = {
        "text_field": "text",
        "label_field": "category",
        "node_id_field": "node_id",
        "neighbour_field": "neighbour"
    }
    args = set_params()
    rewrite_with_config(
        input_path="./dataset/photo/Photo.csv",
        output_path=f"./dataset/photo/Photo_ratio{args.ratio}.csv",
        config=sample_config,
        ratio=args.ratio*args.multiple,
        seed=42
    )
