import os
import shutil
import random

# Set your source and destination directories
src_folder = "test"
dst_folder = "acid_test_remaining/test"

# Create destination folder if it doesn't exist
os.makedirs(dst_folder, exist_ok=True)

# Get all txt files in the source folder
txt_files = [f for f in os.listdir(src_folder) if f.endswith('.txt')]

existing_files = set(os.listdir("acid_test/test"))
print(len(existing_files))
# Filter out files that already exist in the destination folder
available_files = [f for f in txt_files if f not in existing_files]


# Randomly select 1500 txt files
selected_files = random.sample(txt_files, min(10, len(available_files)))

# Copy selected files to the destination folder
for file in selected_files:
    shutil.copy(os.path.join(src_folder, file), os.path.join(dst_folder, file))

print(f"Copied {len(selected_files)} files to {dst_folder}")
