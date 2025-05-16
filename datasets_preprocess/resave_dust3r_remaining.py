import os
import shutil

# Modify these paths
source_folder = "/ssd2/wenyan/3r_benchmark/preprocess/acid_test/test"
destination_folder = "acid_test_remaining/test/"
filename_list_path = "/ssd2/wenyan/3r_benchmark/preprocess/acid_remaining_test.txt"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Load file IDs from the list
with open(filename_list_path, "r") as f:
    file_ids = {line.strip() for line in f if line.strip()}

# Iterate over source folder files and copy matching ones
for file in os.listdir(source_folder):
    file_id, ext = os.path.splitext(file)
    if ext == ".txt" and file_id in file_ids:
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(destination_folder, file)
        shutil.copy2(src_path, dst_path)  # Use shutil.move(...) to move instead of copy

print("Done copying matched files.")
