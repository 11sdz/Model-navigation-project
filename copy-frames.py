import os
import shutil

def copy_matching_png(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    json_files = {f.replace('.json', '') for f in os.listdir(source_dir) if f.endswith('.json')}

    for file in os.listdir(source_dir):
        if file.endswith('.png'):
            file_base = file.replace('.png', '')
            if file_base in json_files:
                shutil.copy(os.path.join(source_dir, file), os.path.join(destination_dir, file))
                print(f"Copied: {file}")

# Example usage
source_directory = "path/to/source"
destination_directory = "path/to/destination"
copy_matching_png(source_directory, destination_directory)
