import pandas as pd
import pickle
import os

def build_sasrec_sequences(parquet_file, output_dir, min_user_count=5, min_item_count=5):
    """
    读取用户交互日志，执行 5-core 过滤与 ID 稠密化，最终构建并保存用户时间序列。
    """
    # 加载 Parquet 格式的合并数据
    print("正在加载 Parquet 数据...")
    df = pd.read_parquet(parquet_file)
    print(f"原始数据总数: {len(df)}")

    # 1. 过滤正向行为 (剔除 plan_to_watch 和 dropped 等未发生或负向特征)
    print("正在过滤有效正向交互行为...")
    valid_status = ['completed', 'watching', 'on_hold']
    df = df[df['status'].isin(valid_status)].copy()
    print(f"保留正向行为后的数据总数: {len(df)}")

    # 2. 迭代式 K-core 过滤 (剔除冷门用户和冷门物品，降低稀疏性)
    print(f"开始执行 {min_user_count}-core 过滤...")
    iteration = 1
    while True:
        user_counts = df['user_id'].value_counts()
        item_counts = df['anime_id'].value_counts()
        
        # 找出满足最低交互次数阈值的用户和物品
        valid_users = user_counts[user_counts >= min_user_count].index
        valid_items = item_counts[item_counts >= min_item_count].index
        
        # 如果所有保留下来的数据都已满足条件，则停止迭代
        if len(valid_users) == len(user_counts) and len(valid_items) == len(item_counts):
            print(f"  在第 {iteration} 次迭代完成 {min_user_count}-core 过滤。")
            break
            
        # 仅保留存在于有效集合中的交互记录
        df = df[df['user_id'].isin(valid_users) & df['anime_id'].isin(valid_items)]
        print(f"  第 {iteration} 次迭代后的数据总数: {len(df)}")
        iteration += 1

    # 3. ID 稠密化映射 (Dense Encoding，必须预留 0 作为 Padding 占位符)
    print("正在进行 ID 稠密化映射...")
    unique_users = df['user_id'].unique()
    unique_items = df['anime_id'].unique()
    
    # 构建映射字典，索引严格从 1 开始
    user2id = {user: i + 1 for i, user in enumerate(unique_users)}
    item2id = {item: i + 1 for i, item in enumerate(unique_items)}
    
    # 映射到新的稠密 ID 列
    df['user_id_encoded'] = df['user_id'].map(user2id)
    df['anime_id_encoded'] = df['anime_id'].map(item2id)
    
    print(f"最终有效用户数: {len(user2id)}, 有效物品数: {len(item2id)}")

    # 4. 按用户分组构建交互序列
    print("正在按用户将物品聚合为时间序列...")
    # 由于数据在上一合并阶段已按时间排序，此处的 groupby 聚合会自然保持时间先后顺序
    sequences_df = df.groupby('user_id_encoded')['anime_id_encoded'].apply(list).reset_index()
    sequences_df.columns = ['user_id', 'item_sequence']

    # 5. 保存序列数据与映射字典
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("正在保存最终序列和字典文件...")
    # 保存序列化特征数据
    sequences_df.to_parquet(os.path.join(output_dir, 'sasrec_sequences.parquet'), index=False)
    
    # 将字典保存为 pickle 格式，以备后续预测时反向查找真实动漫名称
    with open(os.path.join(output_dir, 'user2id.pkl'), 'wb') as f:
        pickle.dump(user2id, f)
    with open(os.path.join(output_dir, 'item2id.pkl'), 'wb') as f:
        pickle.dump(item2id, f)
        
    print("序列构建成功。")
    return sequences_df

# 路径配置参数
PARQUET_FILE = './mal_dataset_clean/merged_interactions.parquet'
FINAL_OUTPUT_DIR = './sasrec_features'

# 执行构建流程
seq_df = build_sasrec_sequences(PARQUET_FILE, FINAL_OUTPUT_DIR)