import os
import json
import time
import re
import pandas as pd
from tqdm import tqdm
import pickle
import httpx
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from volcenginesdkarkruntime import Ark

# ==========================================
# 0. 暴力清洗系统代理残留 (防止环境干扰)
# ==========================================
proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
for key in proxy_keys:
    if key in os.environ:
        del os.environ[key]

# ==========================================
# 1. 双引擎 API 核心配置区 
# ==========================================
# 读取阿里云 Key
with open("api_key_aliyun.txt", "r", encoding="utf-8") as file:
    ALIYUN_API_KEY = file.read().strip() 
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALIYUN_MODEL = "qwen3.5-flash"

# 读取火山引擎 Key
with open("api_key_volc.txt", "r", encoding="utf-8") as file:
    VOLC_API_KEY = file.read().strip()
VOLC_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
# 注意：请确保这是你在火山控制台创建的有效推理接入点 ID (Endpoint)
VOLC_MODEL = "doubao-seed-2-0-lite-260215" 

# 全局超长等待直连客户端
http_client = httpx.Client(trust_env=False, verify=False, timeout=120.0)

# 初始化客户端 A: 阿里云 (兼容 OpenAI 格式)
client_aliyun = OpenAI(
    api_key=ALIYUN_API_KEY,
    base_url=ALIYUN_BASE_URL,
    http_client=http_client
)

# 初始化客户端 B: 火山引擎 (Ark 专属格式)
client_volc = Ark(
    api_key=VOLC_API_KEY,
    base_url=VOLC_BASE_URL,
    http_client=http_client
)

# ==========================================
# 2. 并发与任务调度配置
# ==========================================
BATCH_SIZE = 30
MAX_RETRIES = 5
# 核心：设置最大并发线程数。建议设置为 4 或 6，太高容易触发双边的 429 并发超限报警
MAX_WORKERS = 4 

# 创建一个全局文件锁，确保多线程写入 checkpoint 时不会导致文件损坏
file_lock = threading.Lock()

def extract_json_from_text(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text

def call_llm_api_worker(romaji_list, provider):
    """
    单个线程的任务函数：负责向指定的提供商（阿里云或火山）发起请求
    """
    system_prompt = (
        "你是一个资深的动漫 (ACG) 本地化翻译专家。你的任务是将用户提供的番剧名称准确翻译为中文官方名称或通用惯用译名。\n"
        "严格遵守以下规则：\n"
        "1. 如果是知名动漫，请输出标准中文译名，确保准确。\n"
        "2. 如果是冷门动漫，请根据罗马音或者拼音或者英文名进行合理的意译或音译，不要留空。\n"
        "3. 必须仅返回一个合法的 JSON 字典，Key 为原始文本，Value 为翻译后的中文名。不要附加任何解释文本。\n"
        "4. 如果不是番剧本身带有书名号，非必要不额外添加书名号。"
    )
    
    user_content = json.dumps(romaji_list, ensure_ascii=False)
    
    # 根据调度分配的 provider，选择对应的客户端和模型
    if provider == "aliyun":
        active_client = client_aliyun
        active_model = ALIYUN_MODEL
    else:
        active_client = client_volc
        active_model = VOLC_MODEL
    
    for attempt in range(MAX_RETRIES):
        try:
            # 两个 SDK 都在底层支持原生的 chat.completions 接口
            response = active_client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请翻译以下列表，返回 JSON 字典：\n{user_content}"}
                ],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            clean_json_text = extract_json_from_text(result_text)
            translated_dict = json.loads(clean_json_text)
            
            return translated_dict
            
        except json.JSONDecodeError as je:
            print(f"\n[Warning] {provider.upper()} JSON 解析失败 (尝试 {attempt + 1}/{MAX_RETRIES})")
            time.sleep(2)
        except Exception as e:
            print(f"\n[Warning] {provider.upper()} API 调用异常 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(5)
            
    print(f"[Error] {provider.upper()} 达到最大重试次数，当前批次翻译失败。")
    return {}

def batch_translate_anime_concurrent(csv_path, checkpoint_path="translation_checkpoint.json"):
    print("[System] 正在初始化 [双引擎多线程] 自动化翻译系统...")
    
    df = pd.read_csv(csv_path, usecols=['anime_id', 'title'])
    df = df.dropna(subset=['anime_id', 'title'])
    unique_titles = df['title'].unique().tolist()
    
    translated_db = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                translated_db = json.load(f)
            print(f"[Info] 成功加载历史记录，已翻译 {len(translated_db)} 条。")
        except Exception as e:
            print(f"[Warning] 历史进度文件读取失败: {e}")
            
    pending_titles = [title for title in unique_titles if title not in translated_db]
    print(f"[Info] 本次任务需翻译: {len(pending_titles)} 条。")
    
    if not pending_titles:
        print("[Info] 所有番剧已翻译完毕。")
        return translated_db
        
    batches = [pending_titles[i:i + BATCH_SIZE] for i in range(0, len(pending_titles), BATCH_SIZE)]
    print(f"[System] 划分为 {len(batches)} 个批次，启用 {MAX_WORKERS} 个并发线程...")

    # 启用线程池执行器
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {}
        
        # 轮询调度分配任务：单数批次给阿里云，双数批次给火山引擎
        for idx, batch in enumerate(batches):
            provider = "aliyun" if idx % 2 == 0 else "volc"
            future = executor.submit(call_llm_api_worker, batch, provider)
            future_to_batch[future] = batch
            
        # 使用 tqdm 包装 as_completed，实现并发进度条更新
        for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="双引擎翻译进度"):
            batch_result = future.result()
            
            if batch_result:
                # 获取到结果后，必须加锁写入文件，防止多线程同时修改 json 导致崩溃
                with file_lock:
                    translated_db.update(batch_result)
                    with open(checkpoint_path, 'w', encoding='utf-8') as f:
                        json.dump(translated_db, f, ensure_ascii=False, indent=4)
            else:
                print("\n[Error] 检测到不可恢复的批次失败。")

    print("[System] 多线程翻译流程结束。")
    return translated_db

def generate_final_inference_dict(csv_path, translated_db, output_pkl_path="id2name.pkl"):
    # (此函数代码无需改变，保持原样即可)
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
    print(f"[Success] 推断字典生成完毕: {output_pkl_path} (容量: {len(id2chinese)} 部)")

if __name__ == "__main__":
    csv_file_path = r".\mal_dataset_clean\anime.csv"
    checkpoint_file = "translation_checkpoint.json"
    final_dict_file = "id2name.pkl"
    
    final_translation_db = batch_translate_anime_concurrent(csv_file_path, checkpoint_file)
    if final_translation_db:
        generate_final_inference_dict(csv_file_path, final_translation_db, final_dict_file)