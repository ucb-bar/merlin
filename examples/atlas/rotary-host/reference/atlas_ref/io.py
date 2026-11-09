"""Small array identity and path helpers shared by replay and comparison."""
import hashlib
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(path):
    path = Path(path)
    return path
