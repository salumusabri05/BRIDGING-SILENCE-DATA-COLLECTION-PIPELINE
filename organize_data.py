import os
import shutil

def get_numeric_key(filename):
    # Extracts the leading number from filename to sort numerically.
    # If the filename does not start with a number or is not a number, returns the filename itself.
    name, ext = os.path.splitext(filename)
    try:
        return (0, int(name))
    except ValueError:
        return (1, filename)

def organize_dataset():
    workspace = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(workspace, 'data')
    other_dir = os.path.join(workspace, 'other')
    
    if not os.path.exists(data_dir):
        print(f"Error: data directory not found at {data_dir}")
        return
        
    categories = sorted(os.listdir(data_dir))
    
    print("Starting data organization...")
    print(f"Source directory: {data_dir}")
    print(f"Target directory for excess: {other_dir}")
    print("-" * 50)
    
    total_moved = 0
    
    for cat in categories:
        cat_path = os.path.join(data_dir, cat)
        if not os.path.isdir(cat_path):
            continue
            
        # Get all video files in the category directory
        video_files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv'))]
        
        # Sort video files numerically
        video_files.sort(key=get_numeric_key)
        
        num_files = len(video_files)
        print(f"Category '{cat}': found {num_files} videos")
        
        if num_files <= 16:
            print(f"  -> Has {num_files} <= 16 videos. No action needed.")
            continue
            
        # The first 16 videos are kept
        kept_videos = video_files[:16]
        # The excess videos are moved
        excess_videos = video_files[16:]
        
        # Create target category directory in 'other'
        target_cat_dir = os.path.join(other_dir, cat)
        os.makedirs(target_cat_dir, exist_ok=True)
        
        print(f"  -> Keeping {len(kept_videos)} videos: {kept_videos}")
        print(f"  -> Moving {len(excess_videos)} excess videos to '{os.path.join('other', cat)}'")
        
        for v in excess_videos:
            src_file = os.path.join(cat_path, v)
            dst_file = os.path.join(target_cat_dir, v)
            
            try:
                shutil.move(src_file, dst_file)
                total_moved += 1
            except Exception as e:
                print(f"    Failed to move {v}: {e}")
                
    print("-" * 50)
    print(f"Data organization complete. Total excess videos moved: {total_moved}")

if __name__ == '__main__':
    organize_dataset()
