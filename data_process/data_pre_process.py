import pandas as pd
import os
import csv

def inspect_and_load_datasets(data_folder, output_filename="mal_dataset_probe_log.txt"):
    """
    遍历文件夹中的所有 csv 文件，自动推断分隔符并尝试加载，
    同时将所有的探索输出保存到指定的 txt 文件中。
    """
    print(f"开始探查数据，日志将写入: {output_filename} ...")
    
    # 用 utf-8 编码打开目标 txt 文件（避免中英文或日文特殊字符报错）
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        
        # 遍历文件夹下所有的文件
        for filename in os.listdir(data_folder):
            if not filename.endswith('.csv'):
                continue
                
            file_path = os.path.join(data_folder, filename)
            print(f"--- 正在探索文件: {filename} ---", file=out_file)
            
            # 1. 打印原始文本的前两行，用肉眼直接观察分隔符
            with open(file_path, 'r', encoding='utf-8') as f:
                header_line = f.readline().strip()
                first_data_line = f.readline().strip()
                print(f"【原始表头】: {header_line}", file=out_file)
                print(f"【原始第一行数据】: {first_data_line}", file=out_file)
            
            # 2. 使用 Python 内置的 csv.Sniffer 自动推断分隔符
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    detected_sep = dialect.delimiter
                    print(f"【推断分隔符】: '{detected_sep}' (ASCII码: {ord(detected_sep)})", file=out_file)
            except Exception as e:
                print(f"【推断分隔符失败】: {e}，将尝试常用分隔符", file=out_file)
                detected_sep = ',' # 默认回退到逗号

            # 3. 使用推断出的分隔符用 Pandas 读取数据
            try:
                # 加上 error_bad_lines=False 或 on_bad_lines='skip' 可以防止某一行脏数据导致整个读取崩溃
                df = pd.read_csv(file_path, sep=detected_sep, on_bad_lines='skip', engine='python')
                
                print(f"【成功加载】: 数据集维度 (行数, 列数) = {df.shape}", file=out_file)
                print("【列名列表】:", df.columns.tolist(), file=out_file)
                print("【数据预览】:", file=out_file)
                # 使用 .to_string() 确保 DataFrame 在 txt 文件中格式对齐
                print(df.head(2).to_string(), file=out_file) 
                print("\n" + "="*50 + "\n", file=out_file)
                
            except Exception as e:
                print(f"【Pandas 加载失败】: {e}\n", file=out_file)
                
    print("探查完成！请去工作区查看生成的 txt 文件。")

# 请确保这里的路径是你实际存放 MAL 数据集的文件夹路径
data_directory = './mal_dataset' 
inspect_and_load_datasets(data_directory)