import os
import json
import time
import re
import pandas as pd
from tqdm import tqdm
import pickle
from openai import OpenAI

# ==========================================
# 1. API 核心配置区 
# ==========================================
# 请将 YOUR_API_KEY 替换为你真实获取到的密钥

with open("api_key.txt", "r", encoding="utf-8") as file:
   API_KEY = file.read() 
   #print(API_KEY)

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.5-flash"

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 批处理大小：每次发送给大模型翻译的番剧数量
BATCH_SIZE = 40
# 重试机制：遇到网络请求失败时的最大重试次数
MAX_RETRIES = 5

def extract_json_from_text(text):
    """
    从模型输出的复杂文本中（可能包含 <think> 标签或 Markdown）精准提取 JSON 字典
    """
    # 移除可能存在的 R1 模型思考过程 <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 匹配大括号包裹的 JSON 内容
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text

def call_llm_api(romaji_list):
    """
    调用大语言模型 API，执行批量翻译。
    """
    system_prompt = (
        "你是一个资深的动漫 (ACG) 本地化翻译专家。你的任务是将用户提供的番剧名称准确翻译为中文官方名称或通用惯用译名，并且联网搜索翻译是否准确。\n"
        "严格遵守以下规则：\n"
        "1. 如果是知名动漫，请输出标准中文译名，注意译名必须确保准确。\n"
        "2. 如果是冷门动漫，请根据罗马音或者拼音或者英文名进行合理的意译或音译，不要留空。\n"
        "3. 必须仅返回一个合法的 JSON 字典，Key 为原始罗马音或者拼音或者英文名，Value 为翻译后的中文名。不要附加任何解释文本。\n"
        "4. 如果不是番剧本身带有书名号，非必要不额外添加书名号。"
    )
    
    user_content = json.dumps(romaji_list, ensure_ascii=False)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请翻译以下列表，返回 JSON 字典：\n{user_content}"}
                ],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            
            # 提取并解析 JSON
            clean_json_text = extract_json_from_text(result_text)
            translated_dict = json.loads(clean_json_text)
            
            return translated_dict
            
        except json.JSONDecodeError as je:
            print(f"\n[Warning] JSON 解析失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {je}")
            print(f"原始返回内容片段: {result_text[:100]}...")
            time.sleep(2)
        except Exception as e:
            print(f"\n[Warning] API 调用异常 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(5)
            
    print("[Error] 达到最大重试次数，当前批次翻译失败。")
    return {}

def batch_translate_anime(csv_path, checkpoint_path="translation_checkpoint.json"):
    """
    读取 CSV 文件，执行断点续传的批量翻译
    """
    print("[System] 正在初始化大模型自动化翻译引擎...")
    
    df = pd.read_csv(csv_path, usecols=['anime_id', 'title'])
    df = df.dropna(subset=['anime_id', 'title'])
    
    unique_titles = df['title'].unique().tolist()
    print(f"[Info] 原始数据中共发现 {len(unique_titles)} 个独立番剧名称。")
    
    translated_db = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                translated_db = json.load(f)
            print(f"[Info] 成功加载历史记录，已翻译 {len(translated_db)} 条，继续执行未完成部分。")
        except Exception as e:
            print(f"[Warning] 历史进度文件读取失败: {e}")
            
    pending_titles = [title for title in unique_titles if title not in translated_db]
    print(f"[Info] 本次任务需翻译: {len(pending_titles)} 条。")
    
    if not pending_titles:
        print("[Info] 所有番剧已翻译完毕。")
        return translated_db
        
    batches = [pending_titles[i:i + BATCH_SIZE] for i in range(0, len(pending_titles), BATCH_SIZE)]
    
    print("[System] 开始请求 API...")
    for batch in tqdm(batches, desc="翻译进度"):
        batch_result = call_llm_api(batch)
        
        if batch_result:
            translated_db.update(batch_result)
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(translated_db, f, ensure_ascii=False, indent=4)
        else:
            print("\n[Error] API 连续失败，脚本自动暂停以保护进度。")
            break
            
        time.sleep(1) 
        
    print("[System] 翻译流程结束。")
    return translated_db

def generate_final_inference_dict(csv_path, translated_db, output_pkl_path="id2name.pkl"):
    """
    将翻译数据库与原始 ID 对齐，生成推断字典
    """
    df = pd.read_csv(csv_path, usecols=['anime_id', 'title'])
    df = df.dropna(subset=['anime_id', 'title'])
    df['anime_id'] = df['anime_id'].astype(str)
    
    id2chinese = {}
    missing_count = 0
    
    for _, row in df.iterrows():
        a_id = row['anime_id']
        r_title = row['title']
        
        if r_title in translated_db:
            id2chinese[a_id] = translated_db[r_title]
        else:
            id2chinese[a_id] = r_title
            missing_count += 1
            
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(id2chinese, f)
        
    print("="*60)
    print(f"[Success] 推断字典生成完毕: {output_pkl_path}")
    print(f"  - 字典总容量: {len(id2chinese)} 部番剧")
    print(f"  - 未翻译数量: {missing_count} 部")
    print("="*60)

if __name__ == "__main__":
    csv_file_path = r".\mal_dataset_clean\anime.csv"
    checkpoint_file = "translation_checkpoint.json"
    final_dict_file = "id2name.pkl"
    
    final_translation_db = batch_translate_anime(csv_file_path, checkpoint_file)
    
    if final_translation_db:
        generate_final_inference_dict(csv_file_path, final_translation_db, final_dict_file)