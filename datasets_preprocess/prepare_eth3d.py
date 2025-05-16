import os
import shutil

root_dir = "/ssd2/wenyan/3r_benchmark/datasets/PatchmatchNet/eth3d_high_res_test_subset"

for scene in os.listdir(root_dir):
    scene_path = os.path.join(root_dir, scene)
    if not os.path.isdir(scene_path):
        continue

    pairs_txt_path = os.path.join(scene_path, "pair.txt")
    if not os.path.exists(pairs_txt_path):
        continue

    with open(pairs_txt_path, 'r') as f:
        lines = f.readlines()
        if len(lines) < 3:
            continue
        ref_id = int(lines[1].strip())
        source_line = lines[2].strip().split()
        view_ids = [int(source_line[i]) for i in range(1, len(source_line), 2)]

        keep_ids = set([ref_id] + view_ids)
    print(scene, keep_ids)
    for img_file in os.listdir(os.path.join(scene_path, "images")):
        if img_file.lower().endswith(('.jpg', '.png')):
            img_id = int(os.path.splitext(img_file)[0])
            if img_id not in keep_ids:
                os.remove(os.path.join(scene_path, "images", img_file))
                # print(f"removing {os.path.join(scene_path, "images", img_file)}")
