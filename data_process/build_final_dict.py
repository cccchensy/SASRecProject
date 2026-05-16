import os
import json
import pickle
import pandas as pd

def build_comprehensive_id2name_dict():
    """
    以 item2id.pkl 为绝对基准，交叉对比 anime.csv 和 translation_checkpoint.json，
    生成包含所有模型可用番剧名称的 id2name.pkl 字典。
    """
    print("[System] 正在启动全量字典对齐与融合程序...")
    
    # 配置文件路径
    item2id_path = r"./sasrec_features/item2id.pkl"
    anime_csv_path = r".\mal_dataset_clean\anime.csv"
    translation_json_path = "translation_checkpoint.json"
    output_pkl_path = "id2name.pkl"
    
    # 1. 加载模型的基础 ID 字典
    if not os.path.exists(item2id_path):
        print(f"[Error] 未找到模型字典文件: {item2id_path}")
        return
        
    with open(item2id_path, 'rb') as f:
        item2id = pickle.load(f)
    print(f"[Info] 成功加载 item2id.pkl，共包含 {len(item2id)} 个基础 ID 映射。")
    
    # 2. 加载 anime.csv 获取罗马音底库
    try:
        df = pd.read_csv(anime_csv_path, usecols=['anime_id', 'title'])
        df = df.dropna(subset=['anime_id', 'title'])
        # 构建 Raw ID (str) -> 罗马音 的基础字典
        rawid2romaji = dict(zip(df['anime_id'].astype(str), df['title']))
        print(f"[Info] 成功读取 anime.csv，提取 {len(rawid2romaji)} 条罗马音数据。")
    except Exception as e:
        print(f"[Error] 读取 anime.csv 失败: {e}")
        return
        
    # 3. 加载大模型翻译 JSON 文件
    translated_db = {}
    if os.path.exists(translation_json_path):
        try:
            with open(translation_json_path, 'r', encoding='utf-8') as f:
                translated_db = json.load(f)
            print(f"[Info] 成功加载 translation_checkpoint.json，包含 {len(translated_db)} 条中文翻译。")
        except Exception as e:
            print(f"[Warning] 加载翻译 JSON 失败: {e}。将全部回退为罗马音。")
    else:
        print("[Warning] 未找到翻译 JSON 文件，将全部使用罗马音作为名称。")

    # 4. 执行交叉比对与降级映射
    id2name = {}
    translated_count = 0
    fallback_romaji_count = 0
    missing_csv_count = 0
    
    for raw_id in item2id.keys():
        raw_id_str = str(raw_id)
        
        # 排除作为 Padding 存在的 0 (通常模型的 Padding ID 映射自原始的 0)
        if raw_id_str == '0':
            continue
            
        romaji_title = rawid2romaji.get(raw_id_str)
        
        if romaji_title:
            if romaji_title in translated_db:
                # 优先级 1: 使用翻译好的中文名
                id2name[raw_id_str] = translated_db[romaji_title]
                translated_count += 1
            else:
                # 优先级 2: JSON 中没有翻译，降级使用罗马音
                id2name[raw_id_str] = romaji_title
                fallback_romaji_count += 1
        else:
            # 优先级 3: 极端情况，anime.csv 中甚至查不到这个 ID
            id2name[raw_id_str] = f"未知番剧_{raw_id_str}"
            missing_csv_count += 1

    # 5. 序列化并导出最终字典
    try:
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(id2name, f)
            
        print("\n" + "="*60)
        print(f"[Success] 最终推断字典 {output_pkl_path} 构建完毕！")
        print("="*60)
        print(" 数据统计报告：")
        print(f" - 成功匹配中文名: {translated_count} 部")
        print(f" - 降级使用罗马音: {fallback_romaji_count} 部")
        print(f" - 缺失元数据兜底: {missing_csv_count} 部")
        print(f" - 字典总输出容量: {len(id2name)} 部 (已剔除 Padding ID)")
        print("="*60)
        
    except Exception as e:
        print(f"[Error] 保存 id2name.pkl 失败: {e}")

if __name__ == "__main__":
    build_comprehensive_id2name_dict()