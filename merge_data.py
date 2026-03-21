import os
import glob
import pandas as pd

def merge_and_compress_interactions(clean_data_dir, output_file):
    print("🚀 开始合并并压缩用户交互数据 (保留所有交互类型)...")
    
    file_list = glob.glob(os.path.join(clean_data_dir, 'user_anime*.csv'))
    
    if not file_list:
        print("❌ 没有找到匹配的 user_anime 文件，请检查路径！")
        return None

    # 内存优化：只读取需要的列，极大减轻内存压力
    columns_to_keep = ['user_id', 'anime_id', 'last_interaction_date', 'status']
    df_list = []
    
    for file in file_list:
        try:
            df_chunk = pd.read_csv(file, usecols=columns_to_keep)
            
            # 清洗 1：剔除没有时间戳的脏数据 (SASRec 序列推荐的基石是时间)
            df_chunk = df_chunk.dropna(subset=['last_interaction_date'])
            
            # (取消了所有的状态过滤，原汁原味保留 2.2 亿条交互)
            
            df_list.append(df_chunk)
        except Exception as e:
            print(f"  ❌ 读取 {os.path.basename(file)} 失败: {e}")
            
    print("\n🔄 正在拼接并优化内存占用...")
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"📊 初始交互记录总数 (含 plan_to_watch/dropped 等): {len(full_df)}")
    
    # 内存优化绝招：将包含大量重复字符串的分类字段转换为 category 类型
    if 'status' in full_df.columns:
        full_df['status'] = full_df['status'].astype('category')
    
    # 时间转换与全局排序
    print("⏱️ 正在转换时间并按用户排序 (2.2亿条数据这步极其吃 CPU，请耐心喝杯水)...")
    full_df['last_interaction_date'] = pd.to_datetime(full_df['last_interaction_date'], errors='coerce')
    
    # 清洗 2：丢弃时间格式彻底损坏导致转换出 NaT 的极少数行
    full_df = full_df.dropna(subset=['last_interaction_date']) 
    
    print("⏳ 开始进行排序操作 (保证序列因果性)...")
    full_df = full_df.sort_values(by=['user_id', 'last_interaction_date'])
    
    print(f"📉 最终有效且时序正确的记录数: {len(full_df)}")
    
    # 保存结果
    print(f"💾 正在以 Parquet 极速压缩格式保存到 {output_file} ...")
    # 核心修改：使用 pyarrow 引擎将 2.2 亿行写入单一文件
    full_df.to_parquet(output_file, engine='pyarrow', index=False)
    
    print("🎉 大功告成！不仅免于硬盘爆炸，以后读取这 2.2 亿行数据只需要几秒钟！")
    return full_df

# 注意这里的输出文件后缀变成了 .parquet
CLEAN_DATA_DIR = './mal_dataset_clean'
OUTPUT_FILE = './mal_dataset_clean/merged_interactions.parquet'

merged_df = merge_and_compress_interactions(CLEAN_DATA_DIR, OUTPUT_FILE)