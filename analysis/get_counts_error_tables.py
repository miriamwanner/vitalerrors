import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import json
import statistics

def create_comparison_chart(df, 
                          dataset1='naturalquestions', 
                          dataset2='triviaqa',
                          metric1_col='factscore',
                          metric2_col='vital-precision', 
                          metric3_col='linear-decay-precision',
                          dataset1_label='Natural Questions',
                          dataset2_label='TriviaQA',
                          metric1_label='Factscore',
                          metric2_label='Vital Precision',
                          metric3_label='Linear Decay Precision',
                          chart_title='Comparison Across Datasets and Metrics',
                          # secondary_axis_title='Datasets',
                          figsize=(14, 8),
                          set_alpha=0.6,
                          three_colors=['#1f77b4', '#ff7f0e', '#2ca02c']):
    """
    Create a grouped bar chart comparing three prompt types (normal, missing, wrong) 
    across three metrics for two different dataset groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data with columns: dataset, prompt, and metric columns
    dataset1, dataset2 : str or list of str
        Names of the dataset(s) to compare. Can be single dataset name or list of dataset names.
        If list is provided, values will be averaged across all datasets in the list.
    metric1_col, metric2_col, metric3_col : str
        Column names for the three metrics to compare
    dataset1_label, dataset2_label : str
        Labels for the datasets in the chart
    metric1_label, metric2_label, metric3_label : str
        Labels for the metrics in the chart
    chart_title : str
        Title for the chart
    secondary_axis_title : str
        Title for the secondary x-axis (appears above the dataset group labels)
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Filter and aggregate data for each dataset group
    def get_dataset_metrics(dataset_names, metric_cols):
        # Handle both single dataset name and list of dataset names
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        # Filter data for all specified datasets
        dataset_df = df[df['dataset'].isin(dataset_names)]
        
        # Group by prompt and calculate mean across subsets (val_1, val_2, etc.) and datasets
        grouped = dataset_df.groupby('prompt')[metric_cols].mean()
        
        # Extract values for each prompt type, convert to percentages if needed
        normal_vals = []
        missing_vals = []
        wrong_vals = []
        
        for col in metric_cols:
            # Convert to percentage (multiply by 100) if values are between 0 and 1
            multiplier = 100 if grouped.loc['normal', col] <= 1 else 1
            
            normal_vals.append(grouped.loc['normal', col] * multiplier)
            missing_vals.append(grouped.loc['missing', col] * multiplier)
            wrong_vals.append(grouped.loc['wrong', col] * multiplier)
            
        return normal_vals, missing_vals, wrong_vals
    
    # Get data for both datasets
    metric_cols = [metric1_col, metric2_col, metric3_col]
    
    dataset1_normal, dataset1_missing, dataset1_wrong = get_dataset_metrics(dataset1, metric_cols)
    dataset2_normal, dataset2_missing, dataset2_wrong = get_dataset_metrics(dataset2, metric_cols)
    
    # Combine data for plotting
    all_metrics = [f"{dataset1_label} {metric1_label}", 
                   f"{dataset1_label} {metric2_label}", 
                   f"{dataset1_label} {metric3_label}",
                   f"{dataset2_label} {metric1_label}", 
                   f"{dataset2_label} {metric2_label}", 
                   f"{dataset2_label} {metric3_label}"]
    
    normal_values = dataset1_normal + dataset2_normal
    missing_values = dataset1_missing + dataset2_missing
    wrong_values = dataset1_wrong + dataset2_wrong
    
    # Set up the bar positions
    x = np.arange(len(all_metrics))
    width = 0.25
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the three sets of bars (Normal, Missing, Wrong)
    bars_normal = ax.bar(x - width, normal_values, width, label='Normal', color=three_colors[0], alpha=set_alpha)
    bars_missing = ax.bar(x, missing_values, width, label='Missing', color=three_colors[1], alpha=set_alpha)
    bars_wrong = ax.bar(x + width, wrong_values, width, label='Wrong', color=three_colors[2], alpha=set_alpha)
    
    # Customize the chart
    ax.set_ylabel('Score (%)')
    ax.set_title(chart_title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{metric1_label}", f"{metric2_label}", f"{metric3_label}",
                        f"{metric1_label}", f"{metric2_label}", f"{metric3_label}"], 
                       rotation=45, ha='right')
    ax.legend()
    
    # Set y-axis limits
    all_vals = normal_values + missing_values + wrong_values
    y_max = max(all_vals) * 1.1
    ax.set_ylim(0, y_max)
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    
    # Add vertical line to separate the two datasets
    separation_line_x = 2.5
    ax.axvline(x=separation_line_x, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Create secondary x-axis for dataset labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    # Set up positions for the secondary labels
    group_positions = [1, 4]  # Center positions for each group of 3 metrics
    group_labels = [dataset1_label, dataset2_label]
    ax2.set_xticks(group_positions)
    ax2.set_xticklabels(group_labels, fontweight='bold', fontsize=12)
    ax2.tick_params(axis='x', which='both', length=0)  # Remove tick marks
    
    # Add title for secondary axis
    # ax2.set_xlabel(secondary_axis_title, fontweight='bold', fontsize=14, labelpad=20)
    
    # Position the secondary axis
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='x', pad=60)
    
    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + y_max*0.01,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # add_value_labels(bars_normal)
    # add_value_labels(bars_missing)
    # add_value_labels(bars_wrong)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, ax

def create_comparison_chart_two_bars(df, 
                          dataset1='naturalquestions', 
                          dataset2='triviaqa',
                          metric1_col='factscore',
                          metric2_col='vital-precision', 
                          dataset1_label='Natural Questions',
                          dataset2_label='TriviaQA',
                          metric1_label='Factscore',
                          metric2_label='Vital Precision',
                          chart_title='Comparison Across Datasets and Metrics',
                          # secondary_axis_title='Datasets',
                          figsize=(14, 8),
                          set_alpha=0.6,
                          three_colors=['#1f77b4', '#ff7f0e', '#2ca02c']):
    """
    Create a grouped bar chart comparing three prompt types (normal, missing, wrong) 
    across three metrics for two different dataset groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data with columns: dataset, prompt, and metric columns
    dataset1, dataset2 : str or list of str
        Names of the dataset(s) to compare. Can be single dataset name or list of dataset names.
        If list is provided, values will be averaged across all datasets in the list.
    metric1_col, metric2_col, metric3_col : str
        Column names for the three metrics to compare
    dataset1_label, dataset2_label : str
        Labels for the datasets in the chart
    metric1_label, metric2_label, metric3_label : str
        Labels for the metrics in the chart
    chart_title : str
        Title for the chart
    secondary_axis_title : str
        Title for the secondary x-axis (appears above the dataset group labels)
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Filter and aggregate data for each dataset group
    def get_dataset_metrics(dataset_names, metric_cols):
        # Handle both single dataset name and list of dataset names
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        # Filter data for all specified datasets
        dataset_df = df[df['dataset'].isin(dataset_names)]
        
        # Group by prompt and calculate mean across subsets (val_1, val_2, etc.) and datasets
        grouped = dataset_df.groupby('prompt')[metric_cols].mean()
        
        # Extract values for each prompt type, convert to percentages if needed
        normal_vals = []
        missing_vals = []
        wrong_vals = []
        
        for col in metric_cols:
            # Convert to percentage (multiply by 100) if values are between 0 and 1
            multiplier = 100 if grouped.loc['normal', col] <= 1 else 1
            
            normal_vals.append(grouped.loc['normal', col] * multiplier)
            missing_vals.append(grouped.loc['missing', col] * multiplier)
            wrong_vals.append(grouped.loc['wrong', col] * multiplier)
            
        return normal_vals, missing_vals, wrong_vals
    
    # Get data for both datasets
    metric_cols = [metric1_col, metric2_col]
    
    dataset1_normal, dataset1_missing, dataset1_wrong = get_dataset_metrics(dataset1, metric_cols)
    dataset2_normal, dataset2_missing, dataset2_wrong = get_dataset_metrics(dataset2, metric_cols)

    # put into DataFrame for printing
    table_data = {
        "Dataset": [dataset1_label]*2 + [dataset2_label]*2,
        "Metric": [metric1_label, metric2_label]*2,
        "Normal": dataset1_normal + dataset2_normal,
        "Missing": dataset1_missing + dataset2_missing,
        "Wrong": dataset1_wrong + dataset2_wrong
    }
    result_table = pd.DataFrame(table_data)
    print(result_table)
    
    # Combine data for plotting
    all_metrics = [f"{dataset1_label} {metric1_label}", 
                   f"{dataset1_label} {metric2_label}", 
                   f"{dataset2_label} {metric1_label}", 
                   f"{dataset2_label} {metric2_label}"]
    
    normal_values = dataset1_normal + dataset2_normal
    missing_values = dataset1_missing + dataset2_missing
    wrong_values = dataset1_wrong + dataset2_wrong

    # Set up the bar positions
    x = np.arange(len(all_metrics))
    width = 0.25
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the three sets of bars (Normal, Missing, Wrong)
    bars_normal = ax.bar(x - width, normal_values, width, label='Normal', color=three_colors[0], alpha=set_alpha)
    bars_missing = ax.bar(x, missing_values, width, label='Missing', color=three_colors[1], alpha=set_alpha)
    bars_wrong = ax.bar(x + width, wrong_values, width, label='Wrong', color=three_colors[2], alpha=set_alpha)
    
    # Customize the chart
    ax.set_ylabel('Score (%)')
    ax.set_title(chart_title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{metric1_label}", f"{metric2_label}",
                        f"{metric1_label}", f"{metric2_label}"], 
                       rotation=10, ha='right')
    ax.legend()
    
    # Set y-axis limits
    all_vals = normal_values + missing_values + wrong_values
    y_max = max(all_vals) * 1.1
    ax.set_ylim(0, y_max)
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    
    # Add vertical line to separate the two datasets
    separation_line_x = 1.5
    ax.axvline(x=separation_line_x, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Create secondary x-axis for dataset labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    # Set up positions for the secondary labels
    group_positions = [0.5, 2.5]  # Center positions for each group of 3 metrics
    group_labels = [dataset1_label, dataset2_label]
    ax2.set_xticks(group_positions)
    ax2.set_xticklabels(group_labels, fontweight='bold', fontsize=12)
    ax2.tick_params(axis='x', which='both', length=0)  # Remove tick marks
    
    # Add title for secondary axis
    # ax2.set_xlabel(secondary_axis_title, fontweight='bold', fontsize=14, labelpad=20)
    
    # Position the secondary axis
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='x', pad=60)
    
    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + y_max*0.01,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # add_value_labels(bars_normal)
    # add_value_labels(bars_missing)
    # add_value_labels(bars_wrong)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, ax

def create_comparison_chart_stacked_bars(df, 
                          dataset1='naturalquestions', 
                          dataset2='triviaqa',
                          metric1_col=['factscore', 'vital-precision', 'vital-recall'],
                          metric2_col=['another_metric1', 'another_metric2', 'another_metric3'],
                          dataset1_label='Natural Questions',
                          dataset2_label='TriviaQA',
                          metric1_label = "Subclaims",
                          metric1_labels=['Factscore', 'Vital Precision', 'Vital Recall'],
                          metric2_label = "Nuggets",
                          metric2_labels=['Metric 1', 'Metric 2', 'Metric 3'],
                          chart_title='Comparison Across Datasets and Metrics',
                          figsize=(14, 8),
                          set_alpha=0.6,
                          three_colors=['#1f77b4', '#ff7f0e', '#2ca02c']):
    """
    Create a stacked bar chart comparing three prompt types (normal, missing, wrong) 
    with multiple stacked metrics for two different dataset groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data with columns: dataset, prompt, and metric columns
    dataset1, dataset2 : str or list of str
        Names of the dataset(s) to compare. Can be single dataset name or list of dataset names.
        If list is provided, values will be averaged across all datasets in the list.
    metric1_col, metric2_col : list of str
        Lists of column names for the metrics to stack in each group
    dataset1_label, dataset2_label : str
        Labels for the datasets in the chart
    metric1_labels, metric2_labels : list of str
        Labels for the metrics in each group
    chart_title : str
        Title for the chart
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Ensure metric columns are lists
    if not isinstance(metric1_col, list):
        metric1_col = [metric1_col]
    if not isinstance(metric2_col, list):
        metric2_col = [metric2_col]
    
    # Ensure labels are lists with same length as metric columns
    if not isinstance(metric1_labels, list):
        metric1_labels = [metric1_labels]
    if not isinstance(metric2_labels, list):
        metric2_labels = [metric2_labels]
        
    # Pad labels if needed
    while len(metric1_labels) < len(metric1_col):
        metric1_labels.append(f'Metric {len(metric1_labels) + 1}')
    while len(metric2_labels) < len(metric2_col):
        metric2_labels.append(f'Metric {len(metric2_labels) + 1}')
    
    # Filter and aggregate data for each dataset group
    def get_dataset_metrics(dataset_names, metric_cols):
        # Handle both single dataset name and list of dataset names
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        # Filter data for all specified datasets
        dataset_df = df[df['dataset'].isin(dataset_names)]
        
        # Group by prompt and calculate mean across subsets (val_1, val_2, etc.) and datasets
        grouped = dataset_df.groupby('prompt')[metric_cols].mean()
        
        # Extract values for each prompt type, convert to percentages if needed
        normal_vals = []
        missing_vals = []
        wrong_vals = []
        
        for col in metric_cols:
            # Convert to percentage (multiply by 100) if values are between 0 and 1
            multiplier = 100 if grouped.loc['normal', col] <= 1 else 1
            
            normal_vals.append(grouped.loc['normal', col] * multiplier)
            missing_vals.append(grouped.loc['missing', col] * multiplier)
            wrong_vals.append(grouped.loc['wrong', col] * multiplier)
            
        return normal_vals, missing_vals, wrong_vals
    

    # Get processed tables
    d1_m1 = get_dataset_metrics(dataset1, metric1_col)
    d1_m2 = get_dataset_metrics(dataset1, metric2_col)
    d2_m1 = get_dataset_metrics(dataset2, metric1_col)
    d2_m2 = get_dataset_metrics(dataset2, metric2_col)

    d1_m1.columns = metric1_labels
    d1_m2.columns = metric2_labels
    d2_m1.columns = metric1_labels
    d2_m2.columns = metric2_labels


    # ✅ Print tables before plotting
    print(f"\n=== {dataset1_label} - {', '.join(metric1_labels)} ===")
    print(d1_m1.round(2).to_string())
    print(f"\n=== {dataset1_label} - {', '.join(metric2_labels)} ===")
    print(d1_m2.round(2).to_string())
    print(f"\n=== {dataset2_label} - {', '.join(metric1_labels)} ===")
    print(d2_m1.round(2).to_string())
    print(f"\n=== {dataset2_label} - {', '.join(metric2_labels)} ===")
    print(d2_m2.round(2).to_string())
    
    # Get data for both datasets and both metric groups
    dataset1_metric1_normal, dataset1_metric1_missing, dataset1_metric1_wrong = get_dataset_metrics(dataset1, metric1_col)
    dataset1_metric2_normal, dataset1_metric2_missing, dataset1_metric2_wrong = get_dataset_metrics(dataset1, metric2_col)
    dataset2_metric1_normal, dataset2_metric1_missing, dataset2_metric1_wrong = get_dataset_metrics(dataset2, metric1_col)
    dataset2_metric2_normal, dataset2_metric2_missing, dataset2_metric2_wrong = get_dataset_metrics(dataset2, metric2_col)
    
    # Set up the bar positions
    prompt_types = ['Normal', 'Missing', 'Wrong']
    n_groups = 2  # Two metric groups per dataset
    n_datasets = 2
    n_prompts = 3

    # Create x positions for bars
    group_width = 1.0
    bar_width = 0.25
    dataset_spacing = 2.0
    
    x_positions = []
    x_labels = []
    
    for d, dataset_label in enumerate([dataset1_label, dataset2_label]):
        dataset_offset = d * dataset_spacing
        for g in range(n_groups):
            group_offset = g * group_width
            for p, prompt in enumerate(prompt_types):
                x_pos = dataset_offset + group_offset + p * bar_width
                x_positions.append(x_pos)
                metric_group_label = "Group 1" if g == 0 else "Group 2"
                x_labels.append(f'{prompt}')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Define distinct colors for Normal / Missing / Wrong
    prompt_colors = {
        "Normal": three_colors[0],   # blue
        "Missing": three_colors[1],  # orange
        "Wrong": three_colors[2]     # green
    }

    # Plot bars for each dataset and metric group
    for d, dataset_label in enumerate([dataset1_label, dataset2_label]):
        dataset_offset = d * dataset_spacing

        # Metric Group 1
        group_data = (
            [dataset1_metric1_normal, dataset1_metric1_missing, dataset1_metric1_wrong]
            if d == 0 else
            [dataset2_metric1_normal, dataset2_metric1_missing, dataset2_metric1_wrong]
        )

        for p, prompt in enumerate(prompt_types):
            x_pos = dataset_offset + p * bar_width
            bottom = 0

            for metric_val, metric_label in zip(group_data[p], metric1_labels):
                bar = ax.bar(
                    x_pos, metric_val, bar_width,
                    bottom=bottom,
                    color=prompt_colors[prompt],
                    edgecolor="black", linewidth=0.7,
                    alpha=set_alpha,
                    label=f"{prompt}" if d == 0 and metric_label == metric1_labels[0] else ""
                )

                # Add text just above the bar segment
                ax.text(
                    x_pos, bottom + 0.5,  # x, y position
                    metric_label,                # the label
                    ha='center', va='bottom',    # center horizontally, stick just above
                    fontsize=8, rotation=0      # you can tweak fontsize and rotation
                )

                bottom += metric_val

        # Metric Group 2
        group_data = (
            [dataset1_metric2_normal, dataset1_metric2_missing, dataset1_metric2_wrong]
            if d == 0 else
            [dataset2_metric2_normal, dataset2_metric2_missing, dataset2_metric2_wrong]
        )

        for p, prompt in enumerate(prompt_types):
            x_pos = dataset_offset + group_width + p * bar_width
            bottom = 0

            for metric_val, metric_label in zip(group_data[p], metric1_labels):
                bar = ax.bar(
                    x_pos, metric_val, bar_width,
                    bottom=bottom,
                    color=prompt_colors[prompt],
                    edgecolor="black", linewidth=0.7,
                    alpha=set_alpha,
                    label=f"{prompt}" if d == 0 and metric_label == metric1_labels[0] else ""
                )

                # Add text just above the bar segment
                ax.text(
                    x_pos, bottom + 0.5,  # x, y position
                    metric_label,                # the label
                    ha='center', va='bottom',    # center horizontally, stick just above
                    fontsize=8, rotation=0,      # you can tweak fontsize and rotation
                    # color='white'
                )

                bottom += metric_val

    
    # Customize the chart
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(chart_title, fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis labels
    x_tick_positions = []
    x_tick_labels = []
    
    for d in range(n_datasets):
        dataset_offset = d * dataset_spacing
        # Group 1 positions
        group1_center = dataset_offset + (n_prompts - 1) * bar_width / 2
        # x_tick_positions.extend([dataset_offset + p * bar_width for p in range(n_prompts)])
        x_tick_positions.append(group1_center)
        # Group 2 positions  
        group2_center = dataset_offset + group_width + (n_prompts - 1) * bar_width / 2
        # x_tick_positions.extend([dataset_offset + group_width + p * bar_width for p in range(n_prompts)])
        x_tick_positions.append(group2_center)
    
    x_tick_labels = [f"{metric1_label}", f"{metric2_label}", f"{metric1_label}", f"{metric2_label}"] # prompt_types * 4  # 3 prompts × 2 groups × 2 datasets
    
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, fontsize=9, rotation=45, ha='right')

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, 1))

    # Add vertical lines to separate datasets and groups
    separation_line_x = 1.75
    ax.axvline(x=separation_line_x, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Create secondary x-axis for dataset labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    # Set up positions for the secondary labels
    group_positions = [0.75, 2.75]  # Center positions for each group of 3 metrics
    group_labels = [dataset1_label, dataset2_label]
    ax2.set_xticks(group_positions)
    ax2.set_xticklabels(group_labels, fontweight='bold', fontsize=12)
    ax2.tick_params(axis='x', which='both', length=0)  # Remove tick marks
    
    # Add title for secondary axis
    # ax2.set_xlabel(secondary_axis_title, fontweight='bold', fontsize=14, labelpad=20)
    
    # Position the secondary axis
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='x', pad=60)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, ax

# def print_comparison_data(df, 
#                           dataset1='naturalquestions', 
#                           dataset2='triviaqa',
#                           metric1_col=['factscore', 'vital-precision', 'vital-recall'],
#                           metric2_col=['another_metric1', 'another_metric2', 'another_metric3'],
#                           dataset1_label='Natural Questions',
#                           dataset2_label='TriviaQA',
#                           metric1_label="Subclaims",
#                           metric1_labels=['Factscore', 'Vital Precision', 'Vital Recall'],
#                           metric2_label="Nuggets",
#                           metric2_labels=['Metric 1', 'Metric 2', 'Metric 3']):
#     """
#     Print the data that would be used in the comparison chart.
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         DataFrame containing the data with columns: dataset, prompt, and metric columns
#     dataset1, dataset2 : str or list of str
#         Names of the dataset(s) to compare
#     metric1_col, metric2_col : list of str
#         Lists of column names for the metrics
#     dataset1_label, dataset2_label : str
#         Labels for the datasets
#     metric1_label, metric2_label : str
#         Labels for the metric groups
#     metric1_labels, metric2_labels : list of str
#         Labels for individual metrics
#     """
    
#     # Ensure metric columns are lists
#     if not isinstance(metric1_col, list):
#         metric1_col = [metric1_col]
#     if not isinstance(metric2_col, list):
#         metric2_col = [metric2_col]
    
#     # Ensure labels are lists
#     if not isinstance(metric1_labels, list):
#         metric1_labels = [metric1_labels]
#     if not isinstance(metric2_labels, list):
#         metric2_labels = [metric2_labels]
        
#     # Pad labels if needed
#     while len(metric1_labels) < len(metric1_col):
#         metric1_labels.append(f'Metric {len(metric1_labels) + 1}')
#     while len(metric2_labels) < len(metric2_col):
#         metric2_labels.append(f'Metric {len(metric2_labels) + 1}')
    
#     # Filter and aggregate data for each dataset group
#     def get_dataset_metrics(dataset_names, metric_cols):
#         if isinstance(dataset_names, str):
#             dataset_names = [dataset_names]
        
#         dataset_df = df[df['dataset'].isin(dataset_names)]
#         grouped = dataset_df.groupby('prompt')[metric_cols].mean()
        
#         normal_vals = []
#         missing_vals = []
#         wrong_vals = []
        
#         for col in metric_cols:
#             multiplier = 100 if grouped.loc['normal', col] <= 1 else 1
#             normal_vals.append(grouped.loc['normal', col] * multiplier)
#             missing_vals.append(grouped.loc['missing', col] * multiplier)
#             wrong_vals.append(grouped.loc['wrong', col] * multiplier)
            
#         return normal_vals, missing_vals, wrong_vals
    
#     # Get data for both datasets and both metric groups
#     dataset1_metric1_normal, dataset1_metric1_missing, dataset1_metric1_wrong = get_dataset_metrics(dataset1, metric1_col)
#     dataset1_metric2_normal, dataset1_metric2_missing, dataset1_metric2_wrong = get_dataset_metrics(dataset1, metric2_col)
#     dataset2_metric1_normal, dataset2_metric1_missing, dataset2_metric1_wrong = get_dataset_metrics(dataset2, metric1_col)
#     dataset2_metric2_normal, dataset2_metric2_missing, dataset2_metric2_wrong = get_dataset_metrics(dataset2, metric2_col)
    
#     # Print the data
#     print("=" * 80)
#     print(f"DATA COMPARISON: {dataset1_label} vs {dataset2_label}")
#     print("=" * 80)
    
#     datasets = [
#         (dataset1_label, dataset1_metric1_normal, dataset1_metric1_missing, dataset1_metric1_wrong, 
#          dataset1_metric2_normal, dataset1_metric2_missing, dataset1_metric2_wrong),
#         (dataset2_label, dataset2_metric1_normal, dataset2_metric1_missing, dataset2_metric1_wrong,
#          dataset2_metric2_normal, dataset2_metric2_missing, dataset2_metric2_wrong)
#     ]
    
#     for ds_label, m1_norm, m1_miss, m1_wrong, m2_norm, m2_miss, m2_wrong in datasets:
#         print(f"\n{'=' * 80}")
#         print(f"DATASET: {ds_label}")
#         print(f"{'=' * 80}")
        
#         # Metric Group 1
#         print(f"\n{metric1_label}:")
#         print("-" * 40)
#         for i, label in enumerate(metric1_labels):
#             print(f"\n  {label}:")
#             print(f"    Normal:  {m1_norm[i]:.2f}")
#             print(f"    Missing: {m1_miss[i]:.2f}")
#             print(f"    Wrong:   {m1_wrong[i]:.2f}")
        
#         print(f"\n  Total Stack Heights:")
#         print(f"    Normal:  {sum(m1_norm):.2f}")
#         print(f"    Missing: {sum(m1_miss):.2f}")
#         print(f"    Wrong:   {sum(m1_wrong):.2f}")
        
#         # Metric Group 2
#         print(f"\n{metric2_label}:")
#         print("-" * 40)
#         for i, label in enumerate(metric2_labels):
#             print(f"\n  {label}:")
#             print(f"    Normal:  {m2_norm[i]:.2f}")
#             print(f"    Missing: {m2_miss[i]:.2f}")
#             print(f"    Wrong:   {m2_wrong[i]:.2f}")
        
#         print(f"\n  Total Stack Heights:")
#         print(f"    Normal:  {sum(m2_norm):.2f}")
#         print(f"    Missing: {sum(m2_miss):.2f}")
#         print(f"    Wrong:   {sum(m2_wrong):.2f}")
    
#     print("\n" + "=" * 80)

def print_comparison_data(df, 
                          dataset1='naturalquestions', 
                          dataset2='triviaqa',
                          metric1_col=['factscore', 'vital-precision', 'vital-recall'],
                          metric2_col=['another_metric1', 'another_metric2', 'another_metric3'],
                          dataset1_label='Natural Questions',
                          dataset2_label='TriviaQA',
                          metric1_label="Subclaims",
                          metric1_labels=['Factscore', 'Vital Precision', 'Vital Recall'],
                          metric2_label="Nuggets",
                          metric2_labels=['Metric 1', 'Metric 2', 'Metric 3']):
    """
    Print the data that would be used in the comparison chart.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data with columns: dataset, prompt, and metric columns
    dataset1, dataset2 : str or list of str
        Names of the dataset(s) to compare
    metric1_col, metric2_col : list of str
        Lists of column names for the metrics
    dataset1_label, dataset2_label : str
        Labels for the datasets
    metric1_label, metric2_label : str
        Labels for the metric groups
    metric1_labels, metric2_labels : list of str
        Labels for individual metrics
    """
    
    # Ensure metric columns are lists
    if not isinstance(metric1_col, list):
        metric1_col = [metric1_col]
    if not isinstance(metric2_col, list):
        metric2_col = [metric2_col]
    
    # Ensure labels are lists
    if not isinstance(metric1_labels, list):
        metric1_labels = [metric1_labels]
    if not isinstance(metric2_labels, list):
        metric2_labels = [metric2_labels]
        
    # Pad labels if needed
    while len(metric1_labels) < len(metric1_col):
        metric1_labels.append(f'Metric {len(metric1_labels) + 1}')
    while len(metric2_labels) < len(metric2_col):
        metric2_labels.append(f'Metric {len(metric2_labels) + 1}')
    
    # Filter and aggregate data for each dataset group
    def get_dataset_metrics(dataset_names, metric_cols):
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        dataset_df = df[df['dataset'].isin(dataset_names)]
        grouped = dataset_df.groupby('prompt')[metric_cols].mean()
        
        normal_vals = []
        missing_vals = []
        wrong_vals = []
        
        for col in metric_cols:
            multiplier = 100 if grouped.loc['normal', col] <= 1 else 1
            normal_vals.append(grouped.loc['normal', col] * multiplier)
            missing_vals.append(grouped.loc['missing', col] * multiplier)
            wrong_vals.append(grouped.loc['wrong', col] * multiplier)
            
        return normal_vals, missing_vals, wrong_vals
    
    # Get data for both datasets and both metric groups
    dataset1_metric1_normal, dataset1_metric1_missing, dataset1_metric1_wrong = get_dataset_metrics(dataset1, metric1_col)
    dataset1_metric2_normal, dataset1_metric2_missing, dataset1_metric2_wrong = get_dataset_metrics(dataset1, metric2_col)
    dataset2_metric1_normal, dataset2_metric1_missing, dataset2_metric1_wrong = get_dataset_metrics(dataset2, metric1_col)
    dataset2_metric2_normal, dataset2_metric2_missing, dataset2_metric2_wrong = get_dataset_metrics(dataset2, metric2_col)
    
    # Create table data
    import pandas as pd
    
    table_data = []
    
    datasets = [
        (dataset1_label, dataset1_metric1_normal, dataset1_metric1_missing, dataset1_metric1_wrong, 
         dataset1_metric2_normal, dataset1_metric2_missing, dataset1_metric2_wrong),
        (dataset2_label, dataset2_metric1_normal, dataset2_metric1_missing, dataset2_metric1_wrong,
         dataset2_metric2_normal, dataset2_metric2_missing, dataset2_metric2_wrong)
    ]
    
    for ds_label, m1_norm, m1_miss, m1_wrong, m2_norm, m2_miss, m2_wrong in datasets:
        # Metric Group 1
        for i, label in enumerate(metric1_labels):
            table_data.append({
                'Dataset': ds_label,
                'Metric Group': metric1_label,
                'Metric': label,
                'Normal': f"{m1_norm[i]:.2f}",
                'Missing': f"{m1_miss[i]:.2f}",
                'Wrong': f"{m1_wrong[i]:.2f}"
            })
        
        # Add total for metric group 1
        table_data.append({
            'Dataset': ds_label,
            'Metric Group': metric1_label,
            'Metric': 'TOTAL',
            'Normal': f"{sum(m1_norm):.2f}",
            'Missing': f"{sum(m1_miss):.2f}",
            'Wrong': f"{sum(m1_wrong):.2f}"
        })
        
        # Metric Group 2
        for i, label in enumerate(metric2_labels):
            table_data.append({
                'Dataset': ds_label,
                'Metric Group': metric2_label,
                'Metric': label,
                'Normal': f"{m2_norm[i]:.2f}",
                'Missing': f"{m2_miss[i]:.2f}",
                'Wrong': f"{m2_wrong[i]:.2f}"
            })
        
        # Add total for metric group 2
        table_data.append({
            'Dataset': ds_label,
            'Metric Group': metric2_label,
            'Metric': 'TOTAL',
            'Normal': f"{sum(m2_norm):.2f}",
            'Missing': f"{sum(m2_miss):.2f}",
            'Wrong': f"{sum(m2_wrong):.2f}"
        })
    
    # Create DataFrame and display
    result_df = pd.DataFrame(table_data)
    print(f"\nDATA COMPARISON: {dataset1_label} vs {dataset2_label}")
    print("=" * 100)
    print(result_df.to_string(index=False))
    print("=" * 100)
    
    return result_df


def get_scores(in_file):
    count = 0
    factscores = []
    num_facts = []
    num_supported = []
    with open(in_file, 'r') as file:
        for line in file:
            line = json.loads(line)
            if line["score"]:
                factscores.append(line["score"])
                passage_num_fact = 0
                passage_num_supported = 0
                for decomp_dict in line["decomposition"]:
                    passage_num_fact += len(decomp_dict["decomp"])
                    for atom in decomp_dict["decomp"]:
                        if atom["judgment"] == True:
                            passage_num_supported += 1
                num_facts.append(passage_num_fact)
                num_supported.append(passage_num_supported)
                count += 1
    # factscore stats (precision)
    avg_fs = sum(factscores) / len(factscores)
    avg_n_atom = sum(num_facts) / len(num_facts)
    # safe recall stats (recall)
    median_num_atoms = statistics.median(num_facts)
    recall_scores = [min(a/median_num_atoms, 1) for a in num_supported]
    avg_safe_recall = sum(recall_scores) / len(recall_scores)
    # safe f1 stats
    safe_f1_scores = [(2*prec*rec)/(prec+rec) for prec, rec in zip(factscores, recall_scores)]
    avg_safe_f1 = sum(safe_f1_scores) / len(safe_f1_scores)
    return avg_fs, avg_n_atom, avg_safe_recall, median_num_atoms, avg_safe_f1            


def get_nugget_scores(nuggets_file):
    strict_all_scores = []
    num_nuggets = []
    nuggets_vital = []
    nuggets_okay = []
    num_vital = []
    num_okay = []
    any_vital_nuggets_unsupported = []
    with open(nuggets_file, 'r') as file:
        for line in file:
            line = json.loads(line)
            # strict_all_scores.append(line["response"]["strict-all-score"])
            num_nuggets.append(len(line["nuggets"]))
            vital_total = 0
            vital_support = 0
            okay_total = 0
            okay_support = 0
            for support, n in zip(line["response"]["nugget-assignment"], line["nuggets"]):
                if n["importance"] == "vital":
                    vital_total += 1
                    if support == "support":
                        vital_support += 1
                    elif support == "partial_support":
                        vital_support += 1 # 0.5
                if n["importance"] == "okay":
                    okay_total += 1
                    if support == "support":
                        okay_support += 1
                    elif support == "partial_support":
                        okay_support += 1 # 0.5
            if vital_total == 0:
                nuggets_vital_recall = 0
                nuggets_vital.append(0)
            else:
                nuggets_vital_recall = vital_support / vital_total
                nuggets_vital.append(vital_support / vital_total)
            if okay_total == 0:
                nuggets_okay.append(0)
            else:
                nuggets_okay.append(okay_support / okay_total)
            if nuggets_vital_recall < 1:
                any_vital_nuggets_unsupported.append(1)
            else: 
                any_vital_nuggets_unsupported.append(0)
            num_vital.append(vital_total)
            num_okay.append(okay_total)
            if len(line["nuggets"]) == 0:
                strict_all_scores.append(0)
            else:
                strict_all_scores.append((vital_support+okay_support)/len(line["nuggets"]))
    nuggets_strict_all = sum(strict_all_scores) / len(strict_all_scores)
    avg_n_nuggets = sum(num_nuggets) / len(num_nuggets)
    avg_nuggets_vital = sum(nuggets_vital) / len(nuggets_vital)
    avg_nuggets_okay = sum(nuggets_okay) / len(nuggets_okay)
    avg_num_vital = sum(num_vital) / len(num_vital)
    avg_num_okay = sum(num_okay) / len(num_okay)
    avg_any_vital_nuggets_unsupported = sum(any_vital_nuggets_unsupported) / len(any_vital_nuggets_unsupported)
    return nuggets_strict_all, avg_nuggets_vital, avg_nuggets_okay, avg_n_nuggets, avg_num_vital, avg_num_okay, avg_any_vital_nuggets_unsupported

def get_new_metric_scores(new_metric_file):
    prec_scores = []
    recall_scores = []
    f1_scores = []
    with open(new_metric_file, 'r') as file:
        for line in file:
            line = json.loads(line)
            prec_scores.append(line["weighted-precision"])
            recall_scores.append(line["weighted-recall"])
            f1_scores.append(line["weighted-f1"])
    avg_prec = sum(prec_scores) / len(prec_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    return avg_prec, avg_recall, avg_f1

def get_new_metric_scores(new_metric_file):
    prec_scores = []
    recall_scores = []
    f1_scores = []
    vital_prec = []
    vital_num = []
    okay_prec = []
    okay_num = []
    less_important_prec = []
    less_important_num = []
    linear_decay_precision = []
    linear_decay_recall = []
    linear_decay_f1 = []
    linear_decay_precision_topk = []
    linear_decay_recall_topk = []
    linear_decay_f1_topk = []
    any_vital_wrong = []
    with open(new_metric_file, 'r') as file:
        for line in file:
            line = json.loads(line)
            line = line["scores"]
            prec_scores.append(line["weighted-precision"])
            recall_scores.append(line["weighted-recall"])
            f1_scores.append(line["weighted-f1"])
            vital_prec.append(line["vital-precision"])
            vital_num.append(line["vital-subclaims"])
            okay_prec.append(line["okay-precision"])
            okay_num.append(line["okay-subclaims"])
            less_important_prec.append(line["less-important-precision"])
            less_important_num.append(line["less-important-subclaims"])
            linear_decay_precision.append(line["linear-decay-precision"])
            linear_decay_recall.append(line["linear-decay-recall"])
            linear_decay_f1.append(line["linear-decay-f1"])
            linear_decay_precision_topk.append(line["linear-decay-precision-topk"])
            linear_decay_recall_topk.append(line["linear-decay-recall-topk"])
            linear_decay_f1_topk.append(line["linear-decay-f1-topk"])
            if line["vital-precision"] < 1:
                any_vital_wrong.append(1)
            else: 
                any_vital_wrong.append(0)
    avg_prec = sum(prec_scores) / len(prec_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_vital_prec = sum(vital_prec) / len(vital_prec)
    avg_vital_subclaims = sum(vital_num) / len(vital_num)
    avg_okay_prec = sum(okay_prec) / len(okay_prec)
    avg_okay_subclaims = sum(okay_num) / len(okay_num)
    avg_less_important_prec = sum(less_important_prec) / len(less_important_prec)
    avg_less_important_subclaims = sum(less_important_num) / len(less_important_num)
    avg_linear_decay_precision = sum(linear_decay_precision) / len(linear_decay_precision)
    avg_linear_decay_recall = sum(linear_decay_recall) / len(linear_decay_recall)
    avg_linear_decay_f1 = sum(linear_decay_f1) / len(linear_decay_f1)
    avg_linear_decay_precision_topk = sum(linear_decay_precision_topk) / len(linear_decay_precision_topk)
    avg_linear_decay_recall_topk = sum(linear_decay_recall_topk) / len(linear_decay_recall_topk)
    avg_linear_decay_f1_topk = sum(linear_decay_f1_topk) / len(linear_decay_f1_topk)
    avg_any_vital_wrong = sum(any_vital_wrong) / len(any_vital_wrong)
    return avg_prec, avg_recall, avg_f1, avg_vital_prec, avg_vital_subclaims, avg_okay_prec, avg_okay_subclaims, avg_less_important_prec, avg_less_important_subclaims, avg_linear_decay_precision, avg_linear_decay_recall, avg_linear_decay_f1, avg_linear_decay_precision_topk, avg_linear_decay_recall_topk, avg_linear_decay_f1_topk, avg_any_vital_wrong

def main():
    parser = argparse.ArgumentParser(description='get factscore')
    parser.add_argument('--root_dir', type=str, default="data")
    args = parser.parse_args()
    
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
    
    results = {"dataset": [], "subset": [], "prompt": [], "factscore": [], "num-subclaims": [], "safe-recall": [], "safe-k": [], "safe-f1": [], "nuggets-strict-all": [], "nuggets-vital-recall": [], "nuggets-okay-recall": [], "num-nuggets": [], "num-vital-nuggets": [], "num-okay-nuggets": [], "weighted-precision": [], "weighted-recall": [], "weighted-f1": [], "vital-precision": [], "num-vital-subclaims": [], "okay-precision": [], "num-okay-subclaims": [], "less-important-precision": [], "num-less-important-subclaims": [], "linear-decay-precision": [], "linear-decay-recall": [], "linear-decay-f1": [], "linear-decay-precision-topk": [], "linear-decay-recall-topk": [], "linear-decay-f1-topk": [], "any-vital-wrong": [], "any-vital-nuggets-unsupported": []}
    
    for d in datasets:
        for s in subsets[d]:
            for p in prompts:
                in_file = os.path.join(args.root_dir, d, s, p, "factscore-out.jsonl")
                nuggets_file = os.path.join(args.root_dir, d, s, p, "nuggets-out.jsonl")
                new_metric_file = os.path.join(args.root_dir, d, s, p, "new-metric-out.jsonl")

                avg_fs, avg_n_atom, safe_recall, safe_k, safe_f1 = get_scores(in_file)
                nuggets_strict_all, nuggets_vital, nuggets_okay, avg_n_nuggets, avg_num_vital, avg_num_okay, any_vital_nuggets_unsupported = get_nugget_scores(nuggets_file)
                weighted_prec, weighted_recall, weighted_f1, avg_vital_prec, avg_vital_subclaims, avg_okay_prec, avg_okay_subclaims, avg_less_important_prec, avg_less_important_subclaims, avg_linear_decay_precision, avg_linear_decay_recall, avg_linear_decay_f1, avg_linear_decay_precision_topk, avg_linear_decay_recall_topk, avg_linear_decay_f1_topk, any_vital_wrong = get_new_metric_scores(new_metric_file)

                results["dataset"].append(d)
                results["subset"].append(s)
                results["prompt"].append(p)
                results["factscore"].append(avg_fs)
                results["num-subclaims"].append(avg_n_atom)
                results["safe-recall"].append(safe_recall)
                results["safe-k"].append(safe_k)
                results["safe-f1"].append(safe_f1)
                results["nuggets-strict-all"].append(nuggets_strict_all)
                results["nuggets-vital-recall"].append(nuggets_vital)
                results["nuggets-okay-recall"].append(nuggets_okay)
                results["num-nuggets"].append(avg_n_nuggets)
                results["num-vital-nuggets"].append(avg_num_vital)
                results["num-okay-nuggets"].append(avg_num_okay)
                results["weighted-precision"].append(weighted_prec)
                results["weighted-recall"].append(weighted_recall)
                results["weighted-f1"].append(weighted_f1)
                results["vital-precision"].append(avg_vital_prec)
                results["num-vital-subclaims"].append(avg_vital_subclaims)
                results["okay-precision"].append(avg_okay_prec)
                results["num-okay-subclaims"].append(avg_okay_subclaims)
                results["less-important-precision"].append(avg_less_important_prec)
                results["num-less-important-subclaims"].append(avg_less_important_subclaims)
                results["linear-decay-precision"].append(avg_linear_decay_precision)
                results["linear-decay-recall"].append(avg_linear_decay_recall)
                results["linear-decay-f1"].append(avg_linear_decay_f1)
                results["linear-decay-precision-topk"].append(avg_linear_decay_precision_topk)
                results["linear-decay-recall-topk"].append(avg_linear_decay_recall_topk)
                results["linear-decay-f1-topk"].append(avg_linear_decay_f1_topk)
                results["any-vital-wrong"].append(any_vital_wrong)
                results["any-vital-nuggets-unsupported"].append(any_vital_nuggets_unsupported)
    results_df = pd.DataFrame(results)
    # print(results_df)

    three_colors = ['#FFCC00', '#00A9E0', '#3D5B99']
    set_alpha = 0.9
    dim = (8, 5)
    base_dir = "analysis/counts_errors"


    fig2, ax2 = create_comparison_chart_two_bars(
        results_df,
        dataset1=['factscore', 'wildhallucinations', 'bright'],
        dataset2=['hotpotqa', 'naturalquestions', 'triviaqa'],
        metric1_col='any-vital-wrong',
        metric2_col='any-vital-nuggets-unsupported',
        dataset1_label='Open-Ended', 
        dataset2_label='Single-Answer',
        metric1_label='Any vital subclaims wrong',
        metric2_label='Any vital nuggets unsupported',
        chart_title='Error Metrics',
        # secondary_axis_title='Dataset Group Comparison',
        figsize=dim,
        set_alpha=set_alpha,
        three_colors=three_colors
    )
    # plt.savefig(base_dir + '/errors.pdf', format="pdf")
    #plt.show()

    # fig2, ax2 = create_comparison_chart_stacked_bars(
    #     results_df,
    #     dataset1=['factscore', 'wildhallucinations', 'bright'],
    #     dataset2=['hotpotqa', 'naturalquestions', 'triviaqa'],
    #     metric1_col=['num-vital-subclaims', 'num-okay-subclaims', 'num-less-important-subclaims'],
    #     metric2_col=['num-vital-nuggets', 'num-okay-nuggets'],
    #     dataset1_label='Open-Ended', 
    #     dataset2_label='Single-Answer',
    #     metric1_label="Subclaims",
    #     metric1_labels=['vital', 'okay', 'less'],
    #     metric2_label="Nuggets",
    #     metric2_labels=['vital', 'okay'],
    #     chart_title='Counts',
    #     # secondary_axis_title='Dataset Group Comparison',
    #     figsize=dim,
    #     set_alpha=set_alpha,
    #     three_colors=three_colors
    # )

    count_df = print_comparison_data(
        results_df,
        dataset1=['factscore', 'wildhallucinations', 'bright'],
        dataset2=['hotpotqa', 'naturalquestions', 'triviaqa'],
        metric1_col=['num-vital-subclaims', 'num-okay-subclaims', 'num-less-important-subclaims'],
        metric2_col=['num-vital-nuggets', 'num-okay-nuggets'],
        dataset1_label='Open-Ended', 
        dataset2_label='Single-Answer',
        metric1_label="Subclaims",
        metric1_labels=['vital', 'okay', 'less'],
        metric2_label="Nuggets",
        metric2_labels=['vital', 'okay']
    )
    print(count_df)
    # plt.savefig(base_dir + '/counts.pdf', format="pdf")
    # plt.show()

if __name__ == "__main__":
    main()