import pandas as pd
import json
import pickle
import os

def process_translation_data():
    csv_file_path = r".\mal_dataset_clean\anime.csv"
    json_checkpoint_path = "translation_checkpoint.json"
    untranslated_export_path = "untranslated_anime.csv"
    output_dict_path = "id2name.pkl"

    print("[System] 正在启动数据校验与字典融合程序...")

    # 1. 加载原始 CSV 数据
    try:
        df = pd.read_csv(csv_file_path, usecols=['anime_id', 'title'])
        df = df.dropna(subset=['anime_id', 'title'])
        print(f"[Info] 成功读取原始元数据，共计 {len(df)} 条。")
    except Exception as e:
        print(f"[Error] 读取 anime.csv 失败: {e}")
        return

    # 2. 加载大模型翻译的 JSON 结果
    if os.path.exists(json_checkpoint_path):
        with open(json_checkpoint_path, 'r', encoding='utf-8') as f:
            translated_db = json.load(f)
        print(f"[Info] 成功加载大模型翻译库，共计 {len(translated_db)} 条中文名称。")
    else:
        print(f"[Error] 未找到 {json_checkpoint_path}，请确保翻译脚本已成功运行。")
        return

    # 3. 对比与提取未翻译数据
    untranslated_rows = []
    id2chinese = {}

    for _, row in df.iterrows():
        raw_id = str(row['anime_id'])
        romaji_title = row['title']

        if romaji_title in translated_db:
            id2chinese[raw_id] = translated_db[romaji_title]
        else:
            # 记录未翻译的番剧
            untranslated_rows.append({
                'anime_id': raw_id,
                'title': romaji_title
            })
            # 字典中以原版罗马音作为兜底，防止推断时报错
            id2chinese[raw_id] = romaji_title

    # 4. 导出未翻译的汇总 CSV
    if untranslated_rows:
        untranslated_df = pd.DataFrame(untranslated_rows)
        # 使用 utf-8-sig 编码，确保在 Windows 下用 Excel 打开不会中文乱码
        untranslated_df.to_csv(untranslated_export_path, index=False, encoding='utf-8-sig')
        print(f"\n[Warning] 发现 {len(untranslated_rows)} 部未翻译的番剧！")
        print(f"[Action] 导出至: {untranslated_export_path}")
    else:
        print("\n[Success] 无任何遗漏")

    # 5. 导出最终的推断映射字典
    with open(output_dict_path, 'wb') as f:
        pickle.dump(id2chinese, f)
    
    print("="*60)
    print(f"[Complete] 中文字典已生成: {output_dict_path}")
    print("="*60)

if __name__ == "__main__":
    process_translation_data()