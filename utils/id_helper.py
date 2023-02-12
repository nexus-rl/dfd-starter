import random
import string
from uuid import uuid4 as uuid

alphabet = string.ascii_lowercase + string.digits

def generate_random_id() -> str:
    return uuid().hex
