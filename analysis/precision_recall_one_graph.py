import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import json
import statistics

# def create_linear_decay_chart(df, 
#                               dataset1='naturalquestions', 
#                               dataset2='triviaqa',
#                               dataset1_label='Natural Questions',
#                               dataset2_label='TriviaQA',
#                               chart_title='Linear Decay Metrics',
#                               figsize=(10, 6),
#                               set_alpha=0.6,
#                               three_colors=['#1f77b4', '#ff7f0e', '#2ca02c']):
#     """
#     Create a chart showing only linear decay precision and recall.
#     """
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
    
#     metric_cols = ['linear-decay-precision', 'linear-decay-recall']
    
#     dataset1_normal, dataset1_missing, dataset1_wrong = get_dataset_metrics(dataset1, metric_cols)
#     dataset2_normal, dataset2_missing, dataset2_wrong = get_dataset_metrics(dataset2, metric_cols)
    
#     all_metrics = [f"{dataset1_label}\nLD Precision", 
#                    f"{dataset1_label}\nLD Recall",
#                    f"{dataset2_label}\nLD Precision", 
#                    f"{dataset2_label}\nLD Recall"]
    
#     normal_values = dataset1_normal + dataset2_normal
#     missing_values = dataset1_missing + dataset2_missing
#     wrong_values = dataset1_wrong + dataset2_wrong
    
#     x = np.arange(len(all_metrics))
#     width = 0.25
    
#     fig, ax = plt.subplots(figsize=figsize)
    
#     bars_normal = ax.bar(x - width, normal_values, width, label='Normal', color=three_colors[0], alpha=set_alpha)
#     bars_missing = ax.bar(x, missing_values, width, label='Missing', color=three_colors[1], alpha=set_alpha)
#     bars_wrong = ax.bar(x + width, wrong_values, width, label='Wrong', color=three_colors[2], alpha=set_alpha)
    
#     ax.set_ylabel('Score (%)')
#     ax.set_title(chart_title)
#     ax.set_xticks(x)
#     ax.set_xticklabels(all_metrics, rotation=0, ha='center')
#     ax.legend()
    
#     all_vals = normal_values + missing_values + wrong_values
#     y_max = max(all_vals) * 1.1
#     ax.set_ylim(0, y_max)
    
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    
#     # Separate datasets
#     separation_line_x = 1.5
#     ax.axvline(x=separation_line_x, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
#     ax.grid(True, alpha=0.3, axis='y')
#     plt.tight_layout()
#     return fig, ax


def create_linear_decay_chart(df, 
                              dataset1='naturalquestions', 
                              dataset2='triviaqa',
                              dataset1_label='Natural Questions',
                              dataset2_label='TriviaQA',
                              chart_title='Linear Decay Metrics',
                              figsize=(10, 6),
                              set_alpha=0.6,
                              three_colors=['#1f77b4', '#ff7f0e', '#2ca02c']):
    """
    Create a chart showing only linear decay precision and recall.
    """
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
    
    metric_cols = ['linear-decay-precision', 'linear-decay-recall']
    
    dataset1_normal, dataset1_missing, dataset1_wrong = get_dataset_metrics(dataset1, metric_cols)
    dataset2_normal, dataset2_missing, dataset2_wrong = get_dataset_metrics(dataset2, metric_cols)
    
    all_metrics = [f"{dataset1_label}\nLD Precision", 
                   f"{dataset1_label}\nLD Recall",
                   f"{dataset2_label}\nLD Precision", 
                   f"{dataset2_label}\nLD Recall"]
    
    normal_values = dataset1_normal + dataset2_normal
    missing_values = dataset1_missing + dataset2_missing
    wrong_values = dataset1_wrong + dataset2_wrong
    
    x = np.arange(len(all_metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars_normal = ax.bar(x - width, normal_values, width, label='Normal', color=three_colors[0], alpha=set_alpha)
    bars_missing = ax.bar(x, missing_values, width, label='Missing', color=three_colors[1], alpha=set_alpha)
    bars_wrong = ax.bar(x + width, wrong_values, width, label='Wrong', color=three_colors[2], alpha=set_alpha)
    
    ax.set_ylabel('Score (%)')
    ax.set_title(chart_title)
    ax.set_xticks(x)
    ax.set_xticklabels(['LD Precision', 'LD Recall', 'LD Precision', 'LD Recall'], 
                       rotation=45, ha='right')
    ax.legend()
    
    all_vals = normal_values + missing_values + wrong_values
    y_max = max(all_vals) * 1.1
    ax.set_ylim(0, y_max)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    
    # Separate datasets
    separation_line_x = 1.5
    ax.axvline(x=separation_line_x, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Create secondary x-axis for dataset labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    # Set up positions for the secondary labels
    group_positions = [0.5, 2.5]  # Center positions for each group of 2 metrics
    group_labels = [dataset1_label, dataset2_label]
    ax2.set_xticks(group_positions)
    ax2.set_xticklabels(group_labels, fontweight='bold', fontsize=12)
    ax2.tick_params(axis='x', which='both', length=0)  # Remove tick marks
    
    # Position the secondary axis
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='x', pad=60)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig, ax


# def create_combined_precision_recall_chart(df, 
#                                            dataset1='naturalquestions', 
#                                            dataset2='triviaqa',
#                                            dataset1_label='Natural Questions',
#                                            dataset2_label='TriviaQA',
#                                            chart_title='Precision and Recall Metrics',
#                                            figsize=(14, 6),
#                                            set_alpha=0.6,
#                                            three_colors=['#1f77b4', '#ff7f0e', '#2ca02c']):
#     """
#     Create a chart with Factscore + Vital Precision (precision) and 
#     Nuggets Recall + Vital Recall (recall) with visual separation.
#     """
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
    
#     # Get precision metrics
#     precision_cols = ['factscore', 'vital-precision']
#     dataset1_prec_normal, dataset1_prec_missing, dataset1_prec_wrong = get_dataset_metrics(dataset1, precision_cols)
#     dataset2_prec_normal, dataset2_prec_missing, dataset2_prec_wrong = get_dataset_metrics(dataset2, precision_cols)
    
#     # Get recall metrics
#     recall_cols = ['nuggets-strict-all', 'nuggets-vital-recall']
#     dataset1_rec_normal, dataset1_rec_missing, dataset1_rec_wrong = get_dataset_metrics(dataset1, recall_cols)
#     dataset2_rec_normal, dataset2_rec_missing, dataset2_rec_wrong = get_dataset_metrics(dataset2, recall_cols)
    
#     # Combine all data - create x positions manually to add gap
#     precision_metrics = [
#         f"{dataset1_label}\nFactscore", 
#         f"{dataset1_label}\nVital Prec.",
#         f"{dataset2_label}\nFactscore", 
#         f"{dataset2_label}\nVital Prec."
#     ]
    
#     recall_metrics = [
#         f"{dataset1_label}\nNuggets Rec.",
#         f"{dataset1_label}\nVital Rec.",
#         f"{dataset2_label}\nNuggets Rec.",
#         f"{dataset2_label}\nVital Rec."
#     ]
    
#     # Create x positions with a gap
#     x_precision = np.arange(len(precision_metrics))
#     gap_size = 1.0
#     x_recall = np.arange(len(recall_metrics)) + len(precision_metrics) + gap_size
#     x = np.concatenate([x_precision, x_recall])
#     all_metrics = precision_metrics + recall_metrics
    
#     normal_values = dataset1_prec_normal + dataset2_prec_normal + dataset1_rec_normal + dataset2_rec_normal
#     missing_values = dataset1_prec_missing + dataset2_prec_missing + dataset1_rec_missing + dataset2_rec_missing
#     wrong_values = dataset1_prec_wrong + dataset2_prec_wrong + dataset1_rec_wrong + dataset2_rec_wrong
    
#     width = 0.25
    
#     fig, ax = plt.subplots(figsize=figsize)
    
#     bars_normal = ax.bar(x - width, normal_values, width, label='Normal', color=three_colors[0], alpha=set_alpha)
#     bars_missing = ax.bar(x, missing_values, width, label='Missing', color=three_colors[1], alpha=set_alpha)
#     bars_wrong = ax.bar(x + width, wrong_values, width, label='Wrong', color=three_colors[2], alpha=set_alpha)
    
#     ax.set_ylabel('Score (%)')
#     ax.set_title(chart_title)
#     ax.set_xticks(x)
#     ax.set_xticklabels(all_metrics, rotation=20, ha='right')
#     ax.legend()
    
#     all_vals = [v for v in normal_values + missing_values + wrong_values if v > 0]
#     y_max = max(all_vals) * 1.1
#     ax.set_ylim(0, y_max)
    
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    
#     # Separate datasets within precision and recall
#     ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
#     ax.axvline(x=x_recall[0] + 1.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
#     # Major separation between precision and recall
#     separation_x = (x_precision[-1] + x_recall[0]) / 2
#     ax.axvline(x=separation_x, color='black', linestyle='-', alpha=0.8, linewidth=2)
    
#     # Add text labels for precision and recall sections
#     precision_center = x_precision.mean()
#     recall_center = x_recall.mean()
#     ax.text(precision_center, y_max * 0.95, 'PRECISION', ha='center', fontweight='bold', fontsize=14)
#     ax.text(recall_center, y_max * 0.95, 'RECALL', ha='center', fontweight='bold', fontsize=14)
    
#     ax.grid(True, alpha=0.3, axis='y')
#     plt.tight_layout()
#     return fig, ax

def create_combined_precision_recall_chart(df, 
                                           dataset1='naturalquestions', 
                                           dataset2='triviaqa',
                                           dataset1_label='Natural Questions',
                                           dataset2_label='TriviaQA',
                                           chart_title='Precision and Recall Metrics',
                                           figsize=(14, 6),
                                           set_alpha=0.6,
                                           three_colors=['#1f77b4', '#ff7f0e', '#2ca02c']):
    """
    Create a chart with Factscore + Vital Precision (precision) and 
    Nuggets Recall + Vital Recall (recall) with visual separation.
    """
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
    
    # Get precision metrics
    precision_cols = ['factscore', 'vital-precision']
    dataset1_prec_normal, dataset1_prec_missing, dataset1_prec_wrong = get_dataset_metrics(dataset1, precision_cols)
    dataset2_prec_normal, dataset2_prec_missing, dataset2_prec_wrong = get_dataset_metrics(dataset2, precision_cols)
    
    # Get recall metrics
    recall_cols = ['nuggets-strict-all', 'nuggets-vital-recall']
    dataset1_rec_normal, dataset1_rec_missing, dataset1_rec_wrong = get_dataset_metrics(dataset1, recall_cols)
    dataset2_rec_normal, dataset2_rec_missing, dataset2_rec_wrong = get_dataset_metrics(dataset2, recall_cols)
    
    # Combine all data - create x positions manually to add gap
    precision_metrics = [
        f"{dataset1_label}\nFactscore", 
        f"{dataset1_label}\nVital Prec.",
        f"{dataset2_label}\nFactscore", 
        f"{dataset2_label}\nVital Prec."
    ]
    
    recall_metrics = [
        f"{dataset1_label}\nNuggets Rec.",
        f"{dataset1_label}\nVital Rec.",
        f"{dataset2_label}\nNuggets Rec.",
        f"{dataset2_label}\nVital Rec."
    ]
    
    # # Create x positions with a gap
    # x_precision = np.arange(len(precision_metrics))
    # gap_size = 1.0
    # x_recall = np.arange(len(recall_metrics)) + len(precision_metrics) + gap_size
    # x = np.concatenate([x_precision, x_recall])
    # all_metrics = precision_metrics + recall_metrics

    # Create x positions with minimal gap
    x_precision = np.arange(len(precision_metrics))
    gap_size = 0.4
    x_recall = np.arange(len(recall_metrics)) + len(precision_metrics) + gap_size
    x = np.concatenate([x_precision, x_recall])
    all_metrics = precision_metrics + recall_metrics 
    
    normal_values = dataset1_prec_normal + dataset2_prec_normal + dataset1_rec_normal + dataset2_rec_normal
    missing_values = dataset1_prec_missing + dataset2_prec_missing + dataset1_rec_missing + dataset2_rec_missing
    wrong_values = dataset1_prec_wrong + dataset2_prec_wrong + dataset1_rec_wrong + dataset2_rec_wrong
    
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars_normal = ax.bar(x - width, normal_values, width, label='Normal', color=three_colors[0], alpha=set_alpha)
    bars_missing = ax.bar(x, missing_values, width, label='Missing', color=three_colors[1], alpha=set_alpha)
    bars_wrong = ax.bar(x + width, wrong_values, width, label='Wrong', color=three_colors[2], alpha=set_alpha)
    
    ax.set_ylabel('Score (%)')
    # ax.set_title(chart_title)
    ax.set_xticks(x)
    # Simplified labels - just metric names
    simple_labels = ['Factscore', 'Vital Prec.', 'Factscore', 'Vital Prec.',
                     'Nuggets Rec.', 'Vital Rec.', 'Nuggets Rec.', 'Vital Rec.']
    ax.set_xticklabels(simple_labels, rotation=45, ha='right')
    ax.legend()
    
    all_vals = [v for v in normal_values + missing_values + wrong_values if v > 0]
    y_max = max(all_vals) * 1.1
    ax.set_ylim(0, y_max)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
    
    # Separate datasets within precision and recall
    ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=x_recall[0] + 1.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Major separation between precision and recall
    separation_x = (x_precision[-1] + x_recall[0]) / 2
    ax.axvline(x=separation_x, color='black', linestyle='-', alpha=0.8, linewidth=2)
    
    # Add text labels for precision and recall sections
    precision_center = x_precision.mean()
    recall_center = x_recall.mean()
    ax.text(precision_center, y_max * 1.05, 'PRECISION', ha='center', fontweight='bold', fontsize=14)
    ax.text(recall_center, y_max * 1.05, 'RECALL', ha='center', fontweight='bold', fontsize=14)
    
    # Create secondary x-axis for dataset labels (Open-Ended / Single-Answer)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    # Set up positions for the secondary labels in each section
    # Precision section: Open-Ended and Single-Answer
    prec_group1_center = (x_precision[0] + x_precision[1]) / 2
    prec_group2_center = (x_precision[2] + x_precision[3]) / 2
    # Recall section: Open-Ended and Single-Answer
    rec_group1_center = (x_recall[0] + x_recall[1]) / 2
    rec_group2_center = (x_recall[2] + x_recall[3]) / 2
    
    group_positions = [prec_group1_center, prec_group2_center, rec_group1_center, rec_group2_center]
    group_labels = [dataset1_label, dataset2_label, dataset1_label, dataset2_label]
    
    ax2.set_xticks(group_positions)
    ax2.set_xticklabels(group_labels, fontweight='bold', fontsize=12)
    ax2.tick_params(axis='x', which='both', length=0)  # Remove tick marks
    
    # Position the secondary axis
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='x', pad=60)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig, ax



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

# def get_nugget_scores(nuggets_file):
#     strict_all_scores = []
#     num_nuggets = []
#     nuggets_vital = []
#     nuggets_okay = []
#     num_vital = []
#     num_okay = []
#     any_vital_nuggets_unsupported = []
#     with open(nuggets_file, 'r') as file:
#         for line in file:
#             line = json.loads(line)
#             # strict_all_scores.append(line["response"]["strict-all-score"])
#             num_nuggets.append(len(line["nuggets"]))
#             vital_total = 0
#             vital_support = 0
#             okay_total = 0
#             okay_support = 0
#             for support, n in zip(line["response"]["nugget-assignment"], line["nuggets"]):
#                 if n["importance"] == "vital":
#                     vital_total += 1
#                     if support != "not_support":
#                         vital_support += 1
#                 if n["importance"] == "okay":
#                     okay_total += 1
#                     if support != "not_support":
#                         okay_support += 1
#             if vital_total == 0:
#                 nuggets_vital_recall = 0
#                 nuggets_vital.append(0)
#             else:
#                 nuggets_vital_recall = vital_support / vital_total
#                 nuggets_vital.append(vital_support / vital_total)
#             if okay_total == 0:
#                 nuggets_okay.append(0)
#             else:
#                 nuggets_okay.append(okay_support / okay_total)
#             if nuggets_vital_recall < 1:
#                 any_vital_nuggets_unsupported.append(1)
#             else: 
#                 any_vital_nuggets_unsupported.append(0)
#             num_vital.append(vital_total)
#             num_okay.append(okay_total)
#             if len(line["nuggets"]) == 0:
#                 strict_all_scores.append(0)
#             else:
#                 strict_all_scores.append((vital_support+okay_support)/len(line["nuggets"]))
#     nuggets_strict_all = sum(strict_all_scores) / len(strict_all_scores)
#     avg_n_nuggets = sum(num_nuggets) / len(num_nuggets)
#     avg_nuggets_vital = sum(nuggets_vital) / len(nuggets_vital)
#     avg_nuggets_okay = sum(nuggets_okay) / len(nuggets_okay)
#     avg_num_vital = sum(num_vital) / len(num_vital)
#     avg_num_okay = sum(num_okay) / len(num_okay)
#     avg_any_vital_nuggets_unsupported = sum(any_vital_nuggets_unsupported) / len(any_vital_nuggets_unsupported)
#     return nuggets_strict_all, avg_nuggets_vital, avg_nuggets_okay, avg_n_nuggets, avg_num_vital, avg_num_okay, avg_any_vital_nuggets_unsupported

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
                    # elif support == "partial_support":
                    #     vital_support += 0.5
                if n["importance"] == "okay":
                    okay_total += 1
                    if support == "support":
                        okay_support += 1
                    # elif support == "partial_support":
                    #     okay_support += 0.5
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

    three_colors = ['#FFCC00', '#00A9E0', '#3D5B99']
    set_alpha = 0.9
    dim = (8, 5)
    base_dir = "analysis/figures"

    # Example usage in your main function:
    # Replace the relevant sections with:

    # Graph 1: Linear Decay only
    fig1, ax1 = create_linear_decay_chart(
        results_df,
        dataset1=['factscore', 'wildhallucinations', 'bright'],
        dataset2=['hotpotqa', 'naturalquestions', 'triviaqa'],
        dataset1_label='Open-Ended', 
        dataset2_label='Single-Answer',
        chart_title='Linear Decay Precision and Recall',
        figsize=(10, 5),
        set_alpha=0.9,
        three_colors=['#FFCC00', '#00A9E0', '#3D5B99']
    )
    # plt.show()
    plt.savefig(base_dir + '/linear_decay.pdf', format="pdf")

    # Graph 2: Combined Precision and Recall
    fig2, ax2 = create_combined_precision_recall_chart(
        results_df,
        dataset1=['factscore', 'wildhallucinations', 'bright'],
        dataset2=['hotpotqa', 'naturalquestions', 'triviaqa'],
        dataset1_label='Open-Ended', 
        dataset2_label='Single-Answer',
        chart_title='Precision and Recall Metrics Combined',
        figsize=(9, 5),
        set_alpha=0.9,
        three_colors=['#FFCC00', '#00A9E0', '#3D5B99']
    )
    # plt.show()
    plt.savefig(base_dir + '/precision_recall_combined.pdf', format="pdf")



if __name__ == "__main__":
    main()