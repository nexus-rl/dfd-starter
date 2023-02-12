import random
import string

alphabet = string.ascii_lowercase + string.digits

def generate_random_id(length: int = 8) -> str:
    return ''.join(random.choices(alphabet, k=length))
