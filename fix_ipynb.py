import json
import glob
import os
import re

files_to_check = [
    "data.ipynb",
    "webcam_collection.ipynb",
    "webcamgui_.ipynb",
    "reconstruct.ipynb"
]

def process_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    modified = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            original_source = source
            
            # Change MAX_SEQ_LENGTH
            source = re.sub(r'MAX_SEQ_LENGTH\s*=\s*161', 'MAX_SEQ_LENGTH=120', source)
            
            # Change canvas = np.zeros to np.full
            source = re.sub(
                r'canvas\s*=\s*np\.zeros\(\s*\(\s*HEIGHT\s*,\s*WIDTH\s*,\s*3\s*\)\s*,\s*dtype\s*=\s*np\.uint8\s*\)', 
                r'canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)', 
                source
            )
            
            # Change multi-line canvas=np.zeros in data.ipynb
            source = re.sub(
                r'canvas\s*=\s*np\.zeros\(\s*\(\s*HEIGHT\s*,\s*WIDTH\s*,\s*3\s*\)\s*,\s*dtype\s*=\s*np\.uint8\s*\)',
                r'canvas=np.full((HEIGHT,WIDTH,3), 255, dtype=np.uint8)',
                source, flags=re.MULTILINE
            )
            
            # And another try to match exactly how data.ipynb formats it if it didn't match:
            # "canvas=np.zeros(\n            (HEIGHT,WIDTH,3),\n            dtype=np.uint8\n        )"
            source = re.sub(
                r'canvas\s*=\s*np\.zeros\(\n\s*\(\s*HEIGHT\s*,\s*WIDTH\s*,\s*3\s*\)\s*,\n\s*dtype\s*=\s*np\.uint8\n\s*\)',
                r'canvas=np.full(\n            (HEIGHT,WIDTH,3),\n            255,\n            dtype=np.uint8\n        )',
                source
            )
            
            # Change text colors
            source = re.sub(r'\(\s*255\s*,\s*255\s*,\s*255\s*\)', '(0,0,0)', source)
            
            # If changed, update the cell
            if source != original_source:
                # Need to split back into lines keeping newlines
                lines = [line + '\n' for line in source.split('\n')]
                # remove the last empty newline if it didn't exist
                if source and not source.endswith('\n'):
                    lines[-1] = lines[-1].rstrip('\n')
                elif source and source.endswith('\n'):
                    lines.pop()
                    
                cell['source'] = lines
                modified = True
            
    if modified:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print(f"Fixed {filename}")

for f in files_to_check:
    filepath = os.path.join(r"d:\BRIDGING SILENCE DATA COLLECTION PIPELINE", f)
    if os.path.exists(filepath):
        process_file(filepath)
    else:
        print(f"File not found: {filepath}")
