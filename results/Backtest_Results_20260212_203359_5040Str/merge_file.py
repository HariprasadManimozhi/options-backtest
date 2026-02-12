"""
Merge split file parts back into the original file.
Usage: python merge_file.py
"""

import os

ORIGINAL_NAME = r"Backtest_Results_20260212_203359.xlsx"

def merge_file(original_name):
    part_num = 1
    parts = []
    
    while True:
        part_name = f"{original_name}.part{part_num}"
        if not os.path.exists(part_name):
            break
        parts.append(part_name)
        part_num += 1
    
    if not parts:
        print("No part files found!")
        return
    
    print(f"Found {len(parts)} parts. Merging into {original_name}...\n")
    
    with open(original_name, "wb") as outfile:
        for part_name in parts:
            size = os.path.getsize(part_name) / (1024*1024)
            print(f"  Merging: {part_name} ({size:.2f} MB)")
            with open(part_name, "rb") as part_file:
                outfile.write(part_file.read())
    
    final_size = os.path.getsize(original_name) / (1024*1024)
    print(f"\nDone! Merged file: {original_name} ({final_size:.2f} MB)")

if __name__ == "__main__":
    merge_file(ORIGINAL_NAME)
