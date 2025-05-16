import os

metadata_dir = "/ssd2/wenyan/3r_benchmark/preprocess/acid/metadata"
test_dir = "/ssd2/wenyan/3r_benchmark/preprocess/acid/test"

# List of all <seq_name>.txt files
txt_files = [f for f in os.listdir(metadata_dir) if f.endswith('.txt')]

# List of all folder names in test
test_folders = set(os.listdir(test_dir))

# Filter out txt files that do not have corresponding folder in test
missing_txt_files = [f for f in txt_files if f[:-4] not in test_folders]

# Optional: print or save the result
print(f"{len(missing_txt_files)} metadata files without corresponding folders:")
for f in missing_txt_files:
    # print(f)
    os.remove(os.path.join(metadata_dir, f))
