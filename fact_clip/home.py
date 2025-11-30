import os

def get_project_base():
    """
    Get the project root directory.
    
    fact_clip/home.py -> project root is parent of fact_clip/
    """
    fact_clip_dir = os.path.dirname(os.path.realpath(__file__))
    base = os.path.dirname(fact_clip_dir) + "/"
    return base


if __name__ == "__main__":
    print(get_project_base())
