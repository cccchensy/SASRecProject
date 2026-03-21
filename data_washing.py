import os
import pandas as pd
import csv

def clean_and_standardize_csv(input_folder, output_folder):
    """
    读取带脏数据的 \t 分隔文件，清洗后输出为标准的逗号分隔 CSV
    """
    # 良好的工程习惯：永远不要直接覆盖原始数据文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 已创建清洗后数据存放目录: {output_folder}")

    for filename in os.listdir(input_folder):
        if not filename.endswith('.csv'):
            continue
            
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        print(f"⏳ 正在清洗并转换: {filename} ...")
        
        try:
            # 1. 宽容读取：强制使用 \t 分隔，并跳过导致解析崩溃的脏行
            # 使用 engine='python' 处理脏数据更稳健
            df = pd.read_csv(
                input_path, 
                sep='\t', 
                encoding='utf-8', 
                on_bad_lines='skip', 
                engine='python'
            )
            
            # 2. 标准输出：转换为标准的逗号分隔 CSV
            # quoting=csv.QUOTE_MINIMAL 是魔法所在：
            # 如果某部动漫的名字或简介里本身带有逗号，Pandas 会自动用双引号把它包起来，
            # 比如 Steins;Gate, The Movie 会变成 "Steins;Gate, The Movie"，保证列不会错位。
            df.to_csv(
                output_path, 
                sep=',', 
                index=False, 
                encoding='utf-8', 
                quoting=csv.QUOTE_MINIMAL
            )
            
            print(f"  ✅ 转换成功! 获得干净数据维度: {df.shape}")
            
        except Exception as e:
            print(f"  ❌ 转换失败: {e}")

    print("\n🎉 全部数据清洗完毕！请前往目标文件夹查看。")

# 配置你的输入和输出文件夹路径
RAW_DATA_DIR = './mal_dataset'         # 你刚才探查的原始文件夹
CLEAN_DATA_DIR = './mal_dataset_clean' # 洗干净后的数据存放的新文件夹

clean_and_standardize_csv(RAW_DATA_DIR, CLEAN_DATA_DIR)