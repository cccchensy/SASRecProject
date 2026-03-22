import requests

try:
    print("直连阿里云接口...")
    resp = requests.get("https://dashscope.aliyuncs.com", timeout=5)
    print(f"连接成功！HTTP 状态码: {resp.status_code}")
except Exception as e:
    print(f"\n报错详情:\n{e}")