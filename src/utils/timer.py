import time
from contextlib import contextmanager

@contextmanager
def timer(description: str = "代码块"):
    """
    计时上下文管理器
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{description} 执行时间: {end - start:.4f} 秒")