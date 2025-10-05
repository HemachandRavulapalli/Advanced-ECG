import os

def save_text(obj, path="models/dummy.txt"):
    with open(path, "w") as f:
        f.write(str(obj))

def load_text(path="models/dummy.txt"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None
