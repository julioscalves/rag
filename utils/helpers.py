import numpy as np
import hashlib
import time
import secrets
import string

from functools import wraps

import settings

from utils.logging import logger


def generate_hash_from_file(filepath: str) -> str:
    hash_object = hashlib.sha256()

    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(131072), b""):
                hash_object.update(chunk)

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {filepath}") from exc

    return hash_object.hexdigest()


def generate_hash_from_string(string: str) -> str:
    hash_object = hashlib.sha3_512()
    hash_object.update(string.encode("utf-8"))

    return hash_object.hexdigest()


def normalize(x):
    if np.max(x) - np.min(x) > 0:
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    return x


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not settings.ENABLE_PERF_LOGGING:
            return func(*args, **kwargs)

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        logger.info(f"[PERF] {func.__name__} took {elapsed:.6f} seconds")

        return result

    return wrapper


def generate_random_id(k=10):
    chars = string.ascii_letters + string.digits

    return "".join(secrets.choice(chars) for _ in range(k))
