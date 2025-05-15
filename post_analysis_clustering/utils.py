from functools import wraps
import time 

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

custom_colors = {
    0: '#c542f5',  # purple
    1: '#93f542',  # green
    2: '#c24247',  # red
    3: '#42b0f5',  # blue
    4: '#ffac40',  # orange
    5: '#f54287',  # pink
    6: '#f5f542',  # yellowz
    7: '#42f5c5',  # turquoise
    8: '#c542f5',  # lavender
    9: '#f542c5'   # magenta
}

def get_palette(target_cluster, df):
    unique_classes = sorted(df[target_cluster].unique())

    # Create a mapping: index -> color
    color_list = list(custom_colors.values())
    mapped_colors = {}

    for idx, cls in enumerate(unique_classes):
        color = color_list[idx] if idx < len(color_list) else "#000000"  # fallback to black if out of range
        mapped_colors[cls] = color

    return mapped_colors