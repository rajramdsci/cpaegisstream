import yaml
import os

def create_structure(base_path, structure):
    if isinstance(structure, list):
        for item in structure:
            # Create empty files
            file_path = os.path.join(base_path, item.replace("*", "placeholder"))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'a'):
                os.utime(file_path, None)
    elif isinstance(structure, dict):
        for folder, content in structure.items():
            new_path = os.path.join(base_path, folder)
            os.makedirs(new_path, exist_ok=True)
            create_structure(new_path, content)

with open('structure.yaml', 'r') as f:
    data = yaml.safe_load(f)
    create_structure('.', data)
