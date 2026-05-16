import httpx
from openai import OpenAI

# 1. 读取 API Key
try:
    with open("api_key.txt", "r", encoding="utf-8") as f:
        api_key = f.read().strip()
    print(f"[*] API Key 加载成功，长度: {len(api_key)}")
except Exception as e:
    print(f"[Error] 读取 api_key.txt 失败: {e}")
    exit()

# 2. 初始化客户端 
http_client = httpx.Client(trust_env=False, verify=False, timeout=30.0)

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    http_client=http_client
)

print("\n[*] 正在请求...")

try:
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个测试助手。"},
            {"role": "user", "content": "你好，如果你能收到这条消息，请回复“连接正常”。"}
        ],
        temperature=0.1
    )
    
    print("\n" + "="*40)
    print(" [Success]")
    print(" 模型回复:", response.choices[0].message.content)
    print("="*40 + "\n")
    
except Exception as e:
    print("\n[Error] 错误详情:")
    print(f"异常类型: {type(e).__name__}")
    print(f"异常信息: {str(e)}")