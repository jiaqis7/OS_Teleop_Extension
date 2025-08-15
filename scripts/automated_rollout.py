#!/usr/bin/env python3
"""
Automated rollout script for OS_Teleop_Extension
Handles the three-step process: reset, data collection, and model trigger
"""

import os
import sys
import time
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime


class AutomatedRollout:
    def __init__(self, base_dir="/home/stanford/OS_Teleop_Extension"):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise ValueError(f"Base directory {base_dir} does not exist")
        
    def create_trigger_file(self, filename):
        """Create a trigger file in the base directory"""
        trigger_path = self.base_dir / filename
        trigger_path.touch()
        print(f"Created trigger file: {trigger_path}")
        return trigger_path
    
    def remove_trigger_file(self, filename):
        """Remove a trigger file if it exists"""
        trigger_path = self.base_dir / filename
        if trigger_path.exists():
            trigger_path.unlink()
            print(f"Removed trigger file: {trigger_path}")
    
    def wait_for_file_removal(self, filepath, timeout=300, check_interval=1):
        """Wait for a file to be removed (indicating process completion)"""
        start_time = time.time()
        while filepath.exists():
            if time.time() - start_time > timeout:
                print(f"Timeout waiting for {filepath} to be removed")
                return False
            time.sleep(check_interval)
        return True
    
    def reset_environment(self, wait=True, timeout=60):
        """Step 1: Reset the environment"""
        print("\n=== Step 1: Resetting Environment ===")
        trigger_file = self.create_trigger_file("reset_trigger.txt")
        
        if wait:
            print(f"Waiting for reset to complete (timeout: {timeout}s)...")
            if self.wait_for_file_removal(trigger_file, timeout=timeout):
                print("Reset completed successfully")
            else:
                print("Reset may not have completed within timeout")
        
    def collect_data(self, demo_name="demo_01", wait=True, timeout=300):
        """Step 2: Collect data"""
        print(f"\n=== Step 2: Collecting Data for {demo_name} ===")
        trigger_filename = f"log_trigger_{demo_name}.txt"
        trigger_file = self.create_trigger_file(trigger_filename)
        
        if wait:
            print(f"Waiting for data collection to complete (timeout: {timeout}s)...")
            if self.wait_for_file_removal(trigger_file, timeout=timeout):
                print("Data collection completed successfully")
            else:
                print("Data collection may not have completed within timeout")
    
    def trigger_model(self, wait=True, timeout=300):
        """Step 3: Trigger model execution"""
        print("\n=== Step 3: Triggering Model ===")
        trigger_file = self.create_trigger_file("model_trigger.txt")
        
        if wait:
            print(f"Waiting for model execution to complete (timeout: {timeout}s)...")
            if self.wait_for_file_removal(trigger_file, timeout=timeout):
                print("Model execution completed successfully")
                
                # Check for success file
                success_files = list(self.base_dir.glob("*/success.json"))
                if success_files:
                    print(f"\nFound success files in: {[str(f.parent) for f in success_files]}")
            else:
                print("Model execution may not have completed within timeout")
    
    def run_full_rollout(self, demo_name="demo_01", delays=None):
        """Run the complete rollout process"""
        if delays is None:
            delays = {"after_reset": 2, "after_collect": 2, "after_model": 2}
        
        print(f"\n{'='*50}")
        print(f"Starting Automated Rollout for {demo_name}")
        print(f"{'='*50}")
        
        # Step 1: Reset
        self.reset_environment()
        if delays["after_reset"] > 0:
            print(f"Waiting {delays['after_reset']}s before data collection...")
            time.sleep(delays["after_reset"])
        
        # Step 2: Trigger Data Collection
        print(f"\n=== Step 2: Triggering Data Collection for {demo_name} ===")
        trigger_filename = f"log_trigger_{demo_name}.txt"
        data_trigger_file = self.create_trigger_file(trigger_filename)
        
        # Wait briefly for data collection to start (trigger file will be deleted immediately)
        time.sleep(2)
        
        # Step 3: Immediately trigger Model
        print(f"\n=== Step 3: Triggering Model (immediately after data collection) ===")
        model_trigger_file = self.create_trigger_file("model_trigger.txt")
        
        # Data collection runs for a fixed duration (default 22s from global_cfg.py)
        # Model will wait until data is available, then take over
        print(f"\nData collection running (fixed duration ~22s)...")
        print(f"Model triggered and waiting for data...")
        
        # Wait for model trigger to be consumed (indicates model has started)
        start_time = time.time()
        while model_trigger_file.exists() and (time.time() - start_time) < 30:
            time.sleep(0.5)
        
        if not model_trigger_file.exists():
            print("Model has started execution")
        
        # Check if old success.json exists and get its timestamp
        demo_folder = self.base_dir / demo_name
        success_file = demo_folder / "success.json"
        old_mtime = success_file.stat().st_mtime if success_file.exists() else 0
        
        # Wait for NEW success.json to be created (with newer timestamp)
        print(f"Waiting for model to complete and save results...")
        wait_time = 0
        while wait_time < 300:  # 5 minute timeout
            if success_file.exists() and success_file.stat().st_mtime > old_mtime:
                print(f"Model execution completed - success.json found after {wait_time}s")
                break
            time.sleep(1)
            wait_time += 1
            if wait_time % 10 == 0:
                print(f"  Still waiting... ({wait_time}s elapsed)")
        
        if not success_file.exists() or success_file.stat().st_mtime <= old_mtime:
            print("Warning: Model execution may not have completed (no new success.json found)")
        
        if delays["after_collect"] > 0:
            print(f"Waiting {delays['after_collect']}s before next demo...")
            time.sleep(delays["after_collect"])
        
        print(f"\n{'='*50}")
        print("Rollout Complete!")
        print(f"{'='*50}")
    
    def run_multiple_rollouts(self, num_rollouts, demo_prefix="demo", start_index=1, delays=None):
        """Run multiple rollouts in sequence"""
        print(f"\n{'='*70}")
        print(f"Starting {num_rollouts} Automated Rollouts")
        print(f"{'='*70}")
        
        demo_names = []
        for i in range(num_rollouts):
            demo_name = f"{demo_prefix}_{start_index + i:02d}"
            demo_names.append(demo_name)
            print(f"\n\n=== ROLLOUT {i+1}/{num_rollouts}: {demo_name} ===")
            self.run_full_rollout(demo_name, delays)
            
            if i < num_rollouts - 1:
                print(f"\nWaiting 5s before next rollout...")
                time.sleep(5)
        
        print(f"\n{'='*70}")
        print(f"All {num_rollouts} Rollouts Complete!")
        print(f"{'='*70}")
        
        # Generate success tally for all rollouts
        self.generate_success_tally(demo_names)
    
    def generate_success_tally(self, demo_names):
        """Generate a tally of successes across all demos"""
        tally = {
            "total_demos": len(demo_names),
            "total_success": 0,
            "red_success": 0,
            "green_success": 0,
            "blue_success": 0,
            "demo_results": {}
        }
        
        # Collect data from each demo
        for demo_name in demo_names:
            success_file = self.base_dir / demo_name / "success.json"
            if success_file.exists():
                try:
                    with open(success_file, 'r') as f:
                        data = json.load(f)
                    
                    # Record individual demo results
                    tally["demo_results"][demo_name] = {
                        "success": data.get("success", False),
                        "red_contact": data.get("red_white_contact", False),
                        "green_contact": data.get("green_white_contact", False),
                        "blue_contact": data.get("blue_white_contact", False)
                    }
                    
                    # Update tallies
                    if data.get("success", False):
                        tally["total_success"] += 1
                    if data.get("red_white_contact", False):
                        tally["red_success"] += 1
                    if data.get("green_white_contact", False):
                        tally["green_success"] += 1
                    if data.get("blue_white_contact", False):
                        tally["blue_success"] += 1
                        
                except Exception as e:
                    print(f"Error reading {success_file}: {e}")
                    tally["demo_results"][demo_name] = {"error": str(e)}
            else:
                tally["demo_results"][demo_name] = {"error": "success.json not found"}
        
        # Calculate percentages
        if tally["total_demos"] > 0:
            tally["success_percentage"] = (tally["total_success"] / tally["total_demos"]) * 100
            tally["red_percentage"] = (tally["red_success"] / tally["total_demos"]) * 100
            tally["green_percentage"] = (tally["green_success"] / tally["total_demos"]) * 100
            tally["blue_percentage"] = (tally["blue_success"] / tally["total_demos"]) * 100
        
        # Add timestamp
        tally["timestamp"] = datetime.now().isoformat()
        
        # Save tally file
        tally_filename = f"rollout_tally_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        tally_path = self.base_dir / tally_filename
        with open(tally_path, 'w') as f:
            json.dump(tally, f, indent=4)
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUCCESS TALLY SUMMARY")
        print(f"{'='*60}")
        print(f"Total Demos: {tally['total_demos']}")
        print(f"Total Success (all 3 blocks): {tally['total_success']}/{tally['total_demos']} ({tally.get('success_percentage', 0):.1f}%)")
        print(f"Red Block Success: {tally['red_success']}/{tally['total_demos']} ({tally.get('red_percentage', 0):.1f}%)")
        print(f"Green Block Success: {tally['green_success']}/{tally['total_demos']} ({tally.get('green_percentage', 0):.1f}%)")
        print(f"Blue Block Success: {tally['blue_success']}/{tally['total_demos']} ({tally.get('blue_percentage', 0):.1f}%)")
        print(f"\nDetailed results saved to: {tally_path}")
        print(f"{'='*60}")
        
        return tally_path
    
    def generate_cumulative_tally(self):
        """Generate a tally of ALL demos in the base directory"""
        print("\n=== Generating Cumulative Tally for ALL Demos ===")
        
        # Find all demo folders
        demo_folders = sorted([d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("demo_")])
        demo_names = [d.name for d in demo_folders]
        
        if demo_names:
            print(f"Found {len(demo_names)} demo folders")
            return self.generate_success_tally(demo_names)
        else:
            print("No demo folders found")
            return None


def main():
    parser = argparse.ArgumentParser(description="Automated rollout script for OS_Teleop_Extension")
    parser.add_argument("--demo", default="demo_01", help="Demo name (default: demo_01)")
    parser.add_argument("--num-rollouts", type=int, default=1, help="Number of rollouts to perform")
    parser.add_argument("--start-index", type=int, default=1, help="Starting index for demo names")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for processes to complete")
    parser.add_argument("--reset-only", action="store_true", help="Only perform reset")
    parser.add_argument("--collect-only", action="store_true", help="Only perform data collection")
    parser.add_argument("--model-only", action="store_true", help="Only trigger model")
    parser.add_argument("--delay-after-reset", type=int, default=5, help="Delay after reset (seconds)")
    parser.add_argument("--delay-after-collect", type=int, default=1, help="Delay after data collection (seconds)")
    parser.add_argument("--tally-only", action="store_true", help="Only generate cumulative tally of all demos")
    
    args = parser.parse_args()
    
    # Create rollout manager
    rollout = AutomatedRollout()
    
    # Set up delays
    delays = {
        "after_reset": args.delay_after_reset,
        "after_collect": args.delay_after_collect,
        "after_model": 5
    }
    
    # Execute based on arguments
    if args.tally_only:
        rollout.generate_cumulative_tally()
    elif args.reset_only:
        rollout.reset_environment(wait=not args.no_wait)
    elif args.collect_only:
        rollout.collect_data(args.demo, wait=not args.no_wait)
    elif args.model_only:
        rollout.trigger_model(wait=not args.no_wait)
    elif args.num_rollouts > 1:
        # Extract demo prefix from demo name
        demo_parts = args.demo.split('_')
        demo_prefix = '_'.join(demo_parts[:-1]) if len(demo_parts) > 1 else "demo"
        rollout.run_multiple_rollouts(
            args.num_rollouts, 
            demo_prefix=demo_prefix,
            start_index=args.start_index,
            delays=delays
        )
    else:
        rollout.run_full_rollout(args.demo, delays=delays)


if __name__ == "__main__":
    main()