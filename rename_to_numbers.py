import os
import sys

def rename_files_to_numbers(folder_path="new"):
    # Resolve absolute path
    abs_folder_path = os.path.abspath(folder_path)
    
    if not os.path.exists(abs_folder_path):
        print(f"Error: The folder '{folder_path}' (resolved to '{abs_folder_path}') does not exist.")
        return
        
    if not os.path.isdir(abs_folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return

    # List all entries in the directory
    entries = os.listdir(abs_folder_path)
    
    # Filter out subdirectories; only keep files
    files = [e for e in entries if os.path.isfile(os.path.join(abs_folder_path, e))]
    
    if not files:
        print(f"No files found in '{abs_folder_path}'.")
        return
        
    # Sort files alphabetically to ensure consistent ordering
    # Natural sort would be nice, but simple alphabetical sorting is standard.
    # Let's sort alphabetically.
    files.sort()
    
    print(f"Found {len(files)} files to rename in '{abs_folder_path}'.")
    print("Renaming process starting...")

    # Phase 1: Rename files to unique temporary names to prevent naming collisions.
    # For example, if we have '2.mp4' and want to rename it to '1.mp4' but '1.mp4' already exists.
    temp_mappings = []
    for idx, filename in enumerate(files):
        ext = os.path.splitext(filename)[1]
        old_path = os.path.join(abs_folder_path, filename)
        temp_name = f"__temp_rename_{idx}__{ext}"
        temp_path = os.path.join(abs_folder_path, temp_name)
        
        try:
            os.rename(old_path, temp_path)
            temp_mappings.append((temp_path, idx, ext, filename))
        except Exception as e:
            print(f"Error during temporary rename of '{filename}': {e}")
            # If temporary rename fails, we try to restore what we did to be safe, but let's hope it doesn't.
            # Rollback:
            for t_path, _, _, orig_name in temp_mappings:
                try:
                    os.rename(t_path, os.path.join(abs_folder_path, orig_name))
                except Exception:
                    pass
            return

    # Phase 2: Rename from temporary names to final sequential numbers starting from 1.
    success_count = 0
    for temp_path, idx, ext, orig_name in temp_mappings:
        new_name = f"{idx + 1}{ext}"
        new_path = os.path.join(abs_folder_path, new_name)
        
        try:
            os.rename(temp_path, new_path)
            print(f"Renamed: '{orig_name}' -> '{new_name}'")
            success_count += 1
        except Exception as e:
            print(f"Error during final rename of '{orig_name}' (temp: '{os.path.basename(temp_path)}') to '{new_name}': {e}")
            
    print(f"\nSuccessfully renamed {success_count} out of {len(files)} files in '{folder_path}'.")

if __name__ == "__main__":
    # If a path is provided as a command-line argument, use it; otherwise default to 'new'
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "new"
    rename_files_to_numbers(target_dir)
