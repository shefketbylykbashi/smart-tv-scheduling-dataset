import os
import shutil

src_dir = '...'
dest_dir = '...'

for folder_name in os.listdir(src_dir):
    current_folder = os.path.join(src_dir, folder_name)

    for sub_folder_name in os.listdir(current_folder):
        shutil.copy2(os.path.join(current_folder, sub_folder_name),
                     os.path.join(dest_dir, sub_folder_name))
