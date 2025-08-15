#!/usr/bin/env python3
"""
Quick stats script - shows demo statistics without saving to file
"""

import os
import json
from pathlib import Path

def main():
    # Find all demo folders
    demo_folders = sorted([d for d in Path(".").iterdir() if d.is_dir() and d.name.startswith("demo_")])
    
    if not demo_folders:
        print("No demo folders found!")
        return
    
    # Count successes
    total_demos = 0
    total_success = 0
    red_success = 0
    green_success = 0
    blue_success = 0
    
    for demo_folder in demo_folders:
        success_file = demo_folder / "success.json"
        if not success_file.exists():
            continue
            
        try:
            with open(success_file, 'r') as f:
                data = json.load(f)
            
            total_demos += 1
            
            if data.get("success", False):
                total_success += 1
            if data.get("red_white_contact", False):
                red_success += 1
            if data.get("green_white_contact", False):
                green_success += 1
            if data.get("blue_white_contact", False):
                blue_success += 1
                
        except Exception:
            continue
    
    # Print summary
    print(f"\nDemo Statistics ({total_demos} demos)")
    print("-" * 40)
    print(f"Total Success: {total_success}/{total_demos} ({total_success/total_demos*100:.1f}%)")
    print(f"Red Contact:   {red_success}/{total_demos} ({red_success/total_demos*100:.1f}%)")
    print(f"Green Contact: {green_success}/{total_demos} ({green_success/total_demos*100:.1f}%)")
    print(f"Blue Contact:  {blue_success}/{total_demos} ({blue_success/total_demos*100:.1f}%)")

if __name__ == "__main__":
    main()