#!/usr/bin/env python3
"""
Post-processing script to analyze success.json files from all demo folders
and generate rollout statistics.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def find_demo_folders(base_dir="."):
    """Find all demo folders in the given directory."""
    demo_folders = []
    base_path = Path(base_dir)
    
    # Look for folders matching demo_XX pattern
    for item in sorted(base_path.iterdir()):
        if item.is_dir() and item.name.startswith("demo_"):
            demo_folders.append(item)
    
    return demo_folders


def analyze_success_file(success_path):
    """Analyze a single success.json file and return results."""
    try:
        with open(success_path, 'r') as f:
            data = json.load(f)
        
        result = {
            "success": data.get("success", False),
            "red_contact": data.get("red_white_contact", False),
            "green_contact": data.get("green_white_contact", False),
            "blue_contact": data.get("blue_white_contact", False),
            "timing": data.get("timing", {}),
            "path": str(success_path)  # Convert Path to string
        }
        
        # Extract timing information if available
        if "timing" in data and data["timing"]:
            result["total_duration"] = data["timing"].get("total_duration", None)
            result["success_time"] = data["timing"].get("success_time", None)
            result["first_contact_times"] = data["timing"].get("first_contact_times", {})
        
        return result
        
    except FileNotFoundError:
        return {"error": "success.json not found", "path": str(success_path)}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "path": str(success_path)}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "path": str(success_path)}


def generate_statistics(results):
    """Generate statistics from all demo results."""
    stats = {
        "total_demos": len(results),
        "total_success": 0,
        "red_success": 0,
        "green_success": 0,
        "blue_success": 0,
        "demos_with_errors": 0,
        "timing_stats": {
            "avg_duration": 0,
            "avg_success_time": 0,
            "min_success_time": float('inf'),
            "max_success_time": 0,
            "demos_with_timing": 0
        },
        "demo_results": {}
    }
    
    durations = []
    success_times = []
    
    for demo_name, result in results.items():
        stats["demo_results"][demo_name] = result
        
        if "error" in result:
            stats["demos_with_errors"] += 1
            continue
        
        # Count successes
        if result.get("success", False):
            stats["total_success"] += 1
        if result.get("red_contact", False):
            stats["red_success"] += 1
        if result.get("green_contact", False):
            stats["green_success"] += 1
        if result.get("blue_contact", False):
            stats["blue_success"] += 1
        
        # Collect timing data
        if result.get("total_duration") is not None:
            durations.append(result["total_duration"])
            stats["timing_stats"]["demos_with_timing"] += 1
        
        if result.get("success_time") is not None:
            success_times.append(result["success_time"])
            stats["timing_stats"]["min_success_time"] = min(stats["timing_stats"]["min_success_time"], result["success_time"])
            stats["timing_stats"]["max_success_time"] = max(stats["timing_stats"]["max_success_time"], result["success_time"])
    
    # Calculate percentages
    if stats["total_demos"] > 0:
        valid_demos = stats["total_demos"] - stats["demos_with_errors"]
        if valid_demos > 0:
            stats["success_percentage"] = (stats["total_success"] / valid_demos) * 100
            stats["red_percentage"] = (stats["red_success"] / valid_demos) * 100
            stats["green_percentage"] = (stats["green_success"] / valid_demos) * 100
            stats["blue_percentage"] = (stats["blue_success"] / valid_demos) * 100
    
    # Calculate timing averages
    if durations:
        stats["timing_stats"]["avg_duration"] = sum(durations) / len(durations)
    if success_times:
        stats["timing_stats"]["avg_success_time"] = sum(success_times) / len(success_times)
    else:
        stats["timing_stats"]["min_success_time"] = None
        stats["timing_stats"]["max_success_time"] = None
    
    return stats


def print_summary(stats):
    """Print a formatted summary of the statistics."""
    print("\n" + "="*70)
    print("DEMO ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nTotal Demos Found: {stats['total_demos']}")
    if stats['demos_with_errors'] > 0:
        print(f"Demos with Errors: {stats['demos_with_errors']}")
        valid_demos = stats['total_demos'] - stats['demos_with_errors']
        print(f"Valid Demos: {valid_demos}")
    
    print(f"\n--- Success Rates ---")
    print(f"Total Success (all 3 blocks): {stats['total_success']}/{stats['total_demos'] - stats['demos_with_errors']} ({stats.get('success_percentage', 0):.1f}%)")
    print(f"Red Block Success: {stats['red_success']}/{stats['total_demos'] - stats['demos_with_errors']} ({stats.get('red_percentage', 0):.1f}%)")
    print(f"Green Block Success: {stats['green_success']}/{stats['total_demos'] - stats['demos_with_errors']} ({stats.get('green_percentage', 0):.1f}%)")
    print(f"Blue Block Success: {stats['blue_success']}/{stats['total_demos'] - stats['demos_with_errors']} ({stats.get('blue_percentage', 0):.1f}%)")
    
    if stats['timing_stats']['demos_with_timing'] > 0:
        print(f"\n--- Timing Statistics ---")
        print(f"Demos with Timing Data: {stats['timing_stats']['demos_with_timing']}")
        print(f"Average Demo Duration: {stats['timing_stats']['avg_duration']:.2f}s")
        
        if stats['timing_stats']['avg_success_time'] > 0:
            print(f"Average Time to Success: {stats['timing_stats']['avg_success_time']:.2f}s")
            print(f"Fastest Success: {stats['timing_stats']['min_success_time']:.2f}s")
            print(f"Slowest Success: {stats['timing_stats']['max_success_time']:.2f}s")
    
    # Show failed demos
    failed_demos = []
    error_demos = []
    for demo_name, result in stats['demo_results'].items():
        if "error" in result:
            error_demos.append((demo_name, result['error']))
        elif not result.get('success', False):
            failed_demos.append(demo_name)
    
    if failed_demos:
        print(f"\n--- Failed Demos ({len(failed_demos)}) ---")
        for demo in sorted(failed_demos):
            result = stats['demo_results'][demo]
            blocks = []
            if result.get('red_contact'): blocks.append('Red')
            if result.get('green_contact'): blocks.append('Green')
            if result.get('blue_contact'): blocks.append('Blue')
            print(f"  {demo}: {', '.join(blocks) if blocks else 'No blocks'} made contact")
    
    if error_demos:
        print(f"\n--- Demos with Errors ({len(error_demos)}) ---")
        for demo, error in sorted(error_demos):
            print(f"  {demo}: {error}")
    
    print("\n" + "="*70)


def save_analysis(stats, output_path):
    """Save the analysis to a JSON file."""
    # Add metadata
    stats['timestamp'] = datetime.now().isoformat()
    stats['analysis_type'] = 'post_processing'
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nDetailed analysis saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze demo success files")
    parser.add_argument("--dir", default=".", help="Base directory to search for demos (default: current directory)")
    parser.add_argument("--output", help="Output file for detailed analysis (default: demo_analysis_TIMESTAMP.json)")
    parser.add_argument("--filter", help="Filter demos by name pattern (e.g., 'demo_0*' for demo_01-09)")
    parser.add_argument("--verbose", action="store_true", help="Show individual demo results")
    
    args = parser.parse_args()
    
    # Find all demo folders
    demo_folders = find_demo_folders(args.dir)
    
    if args.filter:
        import fnmatch
        demo_folders = [d for d in demo_folders if fnmatch.fnmatch(d.name, args.filter)]
    
    if not demo_folders:
        print("No demo folders found!")
        return
    
    print(f"Found {len(demo_folders)} demo folders to analyze...")
    
    # Analyze each demo
    results = {}
    for demo_folder in demo_folders:
        success_file = demo_folder / "success.json"
        result = analyze_success_file(success_file)
        results[demo_folder.name] = result
        
        if args.verbose:
            if "error" in result:
                print(f"  {demo_folder.name}: ERROR - {result['error']}")
            else:
                success_str = "SUCCESS" if result['success'] else "FAILED"
                blocks = []
                if result['red_contact']: blocks.append('R')
                if result['green_contact']: blocks.append('G')
                if result['blue_contact']: blocks.append('B')
                print(f"  {demo_folder.name}: {success_str} - Blocks: {'/'.join(blocks) if blocks else 'None'}")
    
    # Generate statistics
    stats = generate_statistics(results)
    
    # Print summary
    print_summary(stats)
    
    # Save detailed analysis
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"demo_analysis_{timestamp}.json"
    
    save_analysis(stats, output_path)


if __name__ == "__main__":
    main()