import shutil
import os
import glob


def copy_folder(source_folder, destination_folder):
    best_epoch_folder_path = find_latest_epoch_folder(source_folder)
    
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Copy only the best_epoch_folder
    destination_best_epoch_folder = os.path.join(destination_folder, os.path.basename(best_epoch_folder_path))
    return shutil.copytree(best_epoch_folder_path, destination_best_epoch_folder)

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Removed folder: {folder_path}")
        
def find_latest_epoch_folder(folder_name):
    current_path = os.getcwd()
    # Find all epoch directories
    print(f"Searching for epoch directories in {folder_name} in {current_path}.....")
    exp_dir_pattern = os.path.join(current_path, folder_name, "epoch_*")
    epoch_dirs = glob.glob(exp_dir_pattern)
    
    if not epoch_dirs:
        print("No epoch directories found.")
        return None

    # Extract epoch numbers and sort numerically
    epoch_dirs.sort(key=lambda x: int(x.split("_")[-1]))
    latest_epoch_dir = epoch_dirs[-1]  # Get the directory with the highest epoch number

    print(f"Latest epoch folder path found: {latest_epoch_dir}")
    return latest_epoch_dir
