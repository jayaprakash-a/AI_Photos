import sys
import os

def check_imports():
    print("Checking imports...")
    try:
        import fastapi
        print("[OK] FastAPI")
        import sqlalchemy
        print("[OK] SQLAlchemy")
        import celery
        print("[OK] Celery")
        import redis
        print("[OK] Redis")
        import cv2
        print("[OK] OpenCV")
        import torch
        print("[OK] PyTorch")
        import PIL
        print("[OK] Pillow")
    except ImportError as e:
        print(f"[FAIL] Missing import: {e}")
        return False
    return True

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    if check_imports():
        print("Environment seems ready!")
    else:
        print("Environment has missing dependencies.")
