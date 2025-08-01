from openai import OpenAI  # ✅ 替换为豆包的 OpenAI SDK 适配器

# ✅ 初始化豆包客户端
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="e69aee0d-8c8f-4e6c-93e7-6d156e0bfaba",
)

def call_doubao_paraphrase(text):
    """调用豆包API进行文本重写"""
    prompt = f'''Please rewrite the following English text. Keep the original meaning but change the expression (e.g., use synonyms or adjust the sentence structure). 
Only output the rewritten text directly, without any additional explanation, formatting, or labels.
Text: {text}'''

    try:
        completion = client.chat.completions.create(
            model="doubao-pro-32k-241215",  # ✅ 使用豆包的模型名称
            messages=[
                {"role": "system", "content": "You are a professional rewriting assistant. Rewrite text with the same meaning using different wording. Output ONLY the rewritten text without any extra characters."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"豆包 API 请求失败: {str(e)}")
        return None
