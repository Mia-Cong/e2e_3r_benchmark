#!/bin/bash

base_dir="/ssd2/wenyan/3r_benchmark/monst3r/data/tum"

# Iterate over all scene folders
for scene in "$base_dir"/*/; do
    if [ -d "$scene" ]; then
        file_count=$(find "$scene/rgb/" -type f | wc -l)
        echo "$(basename "$scene"): $file_count files"
    fi
done