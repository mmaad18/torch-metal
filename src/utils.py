import time


def time_function(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"{func.__name__} took {end - start} seconds")
    return result


def time_function_out(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    diff = end - start
    print(f"{func.__name__} took {diff} seconds")
    return result, diff


def title_print(title: str):
    print(f"\n{'='*10}| {title} |{'='*10}\n")
