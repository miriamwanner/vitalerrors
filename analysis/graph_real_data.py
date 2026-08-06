import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import os

def extract_subclaims_with_judgments(data: Dict[str, Any]) -> List[bool]:
    """
    Extract all subclaims and their judgments in order from the decomposition.
    Returns a list of boolean values indicating if each subclaim is supported.
    """
    subclaims = []
    decomposition = data.get('decomposition', [])
    
    for sentence_data in decomposition:
        subclaim_decomps = sentence_data.get('decomp', [])
        for subclaim in subclaim_decomps:
            judgment = subclaim.get('judgment', False)
            subclaims.append(judgment)
    
    return subclaims

def calculate_cumulative_precision(judgments: List[bool]) -> List[Tuple[int, float]]:
    """
    Calculate cumulative precision at each position.
    Returns list of (position, precision) tuples.
    """
    if not judgments:
        return []
    
    cumulative_precision = []
    supported_count = 0
    
    count = 0
    for i, is_supported in enumerate(judgments, 1):
        if is_supported:
            supported_count += 1
        precision = supported_count / i
        cumulative_precision.append((i, precision))
        count += 1
        if count > 31:
            break
    
    return cumulative_precision

def load_and_process_jsonl(filepath: str) -> List[List[Tuple[int, float]]]:
    """
    Load JSONL file and calculate cumulative precision curves for each entry.
    """
    all_curves = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                data = json.loads(line.strip())
                judgments = extract_subclaims_with_judgments(data)
                
                if judgments:  # Only process entries with subclaims
                    curve = calculate_cumulative_precision(judgments)
                    all_curves.append(curve)
                else:
                    print(f"Warning: No subclaims found in {os.path.basename(filepath)}, line {line_num}")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse {os.path.basename(filepath)}, line {line_num}: {e}")
            except Exception as e:
                print(f"Warning: Error processing {os.path.basename(filepath)}, line {line_num}: {e}")
    
    return all_curves

def average_curves(all_curves: List[List[Tuple[int, float]]]) -> Tuple[List[int], List[float]]:
    """
    Average the precision curves across all entries.
    Returns positions and averaged precisions.
    """
    if not all_curves:
        return [], []
    
    # Find the maximum number of subclaims across all entries
    max_length = max(len(curve) for curve in all_curves)
    
    # Dictionary to store precision values at each position
    position_precisions = defaultdict(list)
    
    # Collect all precision values for each position
    for curve in all_curves:
        for pos, precision in curve:
            position_precisions[pos].append(precision)
    
    # Calculate average precision at each position
    positions = []
    avg_precisions = []
    
    for pos in range(1, max_length + 1):
        if pos in position_precisions:
            positions.append(pos)
            avg_precisions.append(np.mean(position_precisions[pos]))
    
    return positions, avg_precisions

def plot_multiple_files_comparison(file_data: Dict[str, Tuple[List[int], List[float], List[List[Tuple[int, float]]]]]):
    """
    Plot comparison of multiple files on the same graph.
    """
    if not file_data:
        print("No data to plot!")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # fig.suptitle('Cumulative FactScore Precision Comparison', fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (filename, (positions, avg_precisions, all_curves, label)) in enumerate(file_data.items()):
        if positions and avg_precisions:
            color = colors[i % len(colors)]
            # label = os.path.basename(filename)
            
            # Plot the averaged curve
            ax.plot(positions, avg_precisions, color=color, linewidth=3, marker='o', 
                   markersize=4, label=f'{label} (n={len(all_curves)})')
    
    ax.set_xlabel('Subclaim Position', fontsize=12)
    ax.set_ylabel('Cumulative Precision', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_averaged_from_multiple_runs(run_groups: Dict[str, List[str]], title="", save_path=None):
    """
    Take multiple runs for each condition, average them, then plot the averaged results.
    run_groups should be like: {'Condition A': ['file1.jsonl', 'file2.jsonl', 'file3.jsonl'], ...}
    """
    if not run_groups:
        print("No run groups provided!")
        return
    
    condition_averages = {}
    
    # Process each condition
    for condition_name, filepaths in run_groups.items():
        print(f"\nProcessing condition: {condition_name}")
        
        # Collect all curves from all files in this condition
        all_condition_curves = []
        total_entries = 0
        
        for filepath in filepaths:
            try:
                curves = load_and_process_jsonl(filepath)
                all_condition_curves.extend(curves)
                total_entries += len(curves)
                print(f"  Loaded {len(curves)} entries from {os.path.basename(filepath)}")
            except FileNotFoundError:
                print(f"  Warning: File not found: {filepath}")
            except Exception as e:
                print(f"  Warning: Error loading {filepath}: {e}")
        
        if all_condition_curves:
            # Calculate average across all runs for this condition
            positions, avg_precisions = average_curves(all_condition_curves)
            condition_averages[condition_name] = (positions, avg_precisions, total_entries)
            print(f"  Total entries for {condition_name}: {total_entries}")
        else:
            print(f"  No valid data found for condition: {condition_name}")
    
    # Plot the condition averages
    if not condition_averages:
        print("No valid condition data to plot!")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # fig.suptitle(f'Cumulative Factscore Precision: {title}', 
    #              fontsize=16, fontweight='bold')
    
    colors = ['#FFCC00', '#00A9E0', '#3D5B99'] # ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (condition_name, (positions, avg_precisions, total_entries)) in enumerate(condition_averages.items()):
        if positions and avg_precisions:
            color = colors[i % len(colors)]

            # Convert precision values to percentages
            avg_precisions_pct = [p * 100 for p in avg_precisions]
            
            # Plot the averaged curve
            ax.plot(positions, avg_precisions_pct, color=color, linewidth=3, marker='o', 
                   markersize=4, label=f'{condition_name}') # (n={total_entries} total entries)')
    
    ax.set_xlabel('Subclaim Position', fontsize=12)
    ax.set_ylabel('Cumulative Precision (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf')
    plt.show()
    
    # # Print summary statistics
    # print(f"\n{'='*80}")
    # print(f"CONDITION COMPARISON SUMMARY")
    # print(f"{'='*80}")
    
    # for condition_name, (positions, avg_precisions, total_entries) in condition_averages.items():
    #     print(f"\n{condition_name}:")
    #     print(f"  Total entries: {total_entries}")
    #     print(f"  Max subclaims: {max(positions) if positions else 0}")
    #     print(f"  Initial precision: {avg_precisions[0]:.4f}" if avg_precisions else "N/A")
    #     print(f"  Final precision: {avg_precisions[-1]:.4f}" if avg_precisions else "N/A")
        
    #     # Show precision at key positions
    #     key_positions = [1, 5, 10, 20]
    #     for key_pos in key_positions:
    #         if key_pos <= len(avg_precisions):
    #             print(f"  Position {key_pos:2d}: {avg_precisions[key_pos-1]:.4f}")

def main():
    """
    Average multiple runs per condition (normal/missing/wrong), then compare
    open-ended vs. single-answer query cumulative precision (Figure 3/4).
    """
    print("=" * 50)
    
    run_groups = {"normal": [], "missing": [], "wrong": []}
    datasets = ["hotpotqa", "naturalquestions", "triviaqa", "bright", "wildhallucinations", "factscore"]
    subsets = {
        "bright": [
            "biology",
            "earth_science",
            "economics",
            "psychology",
            "robotics",
            "stackoverflow",
            "sustainable_living",
        ],
        "wildhallucinations": ["cult_ent_1", "cult_ent_2", "cult_ent_3", "cult_ent_4", "geographic"],
        "factscore": ["bios"],
        "hotpotqa": ["val_1", "val_2"],
        "naturalquestions": ["val_1", "val_2"],
        "triviaqa": ["rc_1", "rc_2"]
    }
    prompts = ["normal", "missing", "wrong"]

    dataset_types = {"open-ended": ["factscore", "wildhallucinations", "bright"], 
                     "single-answer": ["hotpotqa", "naturalquestions", "triviaqa"]}
    root_dir = "data"

    for d in dataset_types["open-ended"]:
        for s in subsets[d]:
            for p in prompts:
                run_groups[p].append(os.path.join(root_dir, d, s, p, "factscore-out.jsonl"))
    title = "open-ended"
    plot_averaged_from_multiple_runs(run_groups, title, 'analysis/figures/fs_oe_no_title.pdf')

    run_groups = {"normal": [], "missing": [], "wrong": []}
    for d in dataset_types["single-answer"]:
        for s in subsets[d]:
            for p in prompts:
                run_groups[p].append(os.path.join(root_dir, d, s, p, "factscore-out.jsonl"))
    title = "single-answer"
    plot_averaged_from_multiple_runs(run_groups, title, 'analysis/figures/fs_sa_no_title.pdf')

if __name__ == "__main__":
    main()