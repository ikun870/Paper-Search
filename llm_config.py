"""
LLM 统一配置模块
所有 LLM 相关的配置集中在此管理，修改 .env 即可切换模型，无需改动业务代码
内置限流重试机制，遇到 429 错误自动等待后重试
"""
import os
import time
from openai import OpenAI

# 延迟初始化，确保 load_dotenv() 已执行
_client = None
# 连续 LLM 调用之间的间隔（秒），防止触发速率限制
_CALL_INTERVAL = 2


def get_client() -> OpenAI:
    """获取 LLM 客户端（单例），配置从 .env 读取

    .env 配置项：
        LLM_API_KEY   - API 密钥（必填）
        LLM_BASE_URL  - API 地址（必填）
        LLM_MODEL     - 模型名称（必填）
    """
    global _client
    if _client is None:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        if not api_key or not base_url:
            raise ValueError("请在 .env 中配置 LLM_API_KEY 和 LLM_BASE_URL")
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def get_model() -> str:
    """获取当前配置的模型名称"""
    model = os.getenv("LLM_MODEL")
    if not model:
        raise ValueError("请在 .env 中配置 LLM_MODEL")
    return model


def chat_with_retry(messages: list, temperature: float = 0.3, max_tokens: int = 500,
                    response_format: dict = None, max_retries: int = 3) -> str:
    """带限流重试的 LLM 调用，遇到 429 错误自动等待后重试

    Args:
        messages: 对话消息列表
        temperature: 生成温度
        max_tokens: 最大输出 token 数
        response_format: 响应格式（如 {"type": "json_object"}）
        max_retries: 最大重试次数

    Returns:
        LLM 返回的文本内容
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            time.sleep(_CALL_INTERVAL)
            kwargs = dict(
                model=get_model(),
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )
            if response_format:
                kwargs["response_format"] = response_format
            resp = get_client().chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            last_error = e
            if "429" in error_str:
                wait = _CALL_INTERVAL * (2 ** attempt)
                print(f"  [LLM] 限流，{wait}s 后重试（第{attempt+1}次）...")
                time.sleep(wait)
            else:
                raise
    raise last_error
