from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    cohen_kappa_score
import re
import hashlib
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans


model_name_corrections = {
    'gpt-3.5-turbo-0125': 'gpt-3.5-turbo-0125',
    'gpt-4o-mini-2024-07-18': 'gpt-4o-mini-2024-07-18',
    'gpt-4o-2024-05-13': 'gpt-4o-2024-05-13',
    'gpt-4o-2024-11-20': 'gpt-4o-2024-11-20',

    "gemini-2.0-flash" : "gemini-2.0-flash",
    "gemini-2.0-flash-lite" : "gemini-2.0-flash-lite",
    "gemini-1.5-pro" : "gemini-1.5-pro",
    "gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
    "gemini-1.5-flash": "gemini-1.5-flash",

    'deepseek-r1_14b': 'deepseek-r1:14b',
    'deepseek-r1_70b': 'deepseek-r1:70b',
    'finalend_athene-70b_latest': 'finalend/athene-70b:latest',
    'gemma2_27b': 'gemma2:27b',
    'gemma2_9b': 'gemma2:9b',
    'gemma_latest': 'gemma:latest',
    'granite3-dense_8b': 'granite3-dense:8b',
    'granite3-moe_latest': 'granite3-moe:latest',
    'granite3.1-dense_2b': 'granite3.1-dense:2b',
    'granite3.1-dense_8b': 'granite3.1-dense:8b',
    'llama3-groq-tool-use_70b': 'llama3-groq-tool-use:70b',
    'llama3-groq-tool-use_8b': 'llama3-groq-tool-use:8b',
    'llama3.1_70b': 'llama3.1:70b',
    'llama3.1_8b': 'llama3.1:8b',
    'llama3.2_1b': 'llama3.2:1b',
    'llama3.2_latest': 'llama3.2:latest',
    'llama3.3_70b': 'llama3.3:70b',
    'llama3_8b-instruct-fp16': 'llama3:8b-instruct-fp16',
    'llama3_latest': 'llama3:latest',
    'meditron_70b': 'meditron:70b',
    'medllama2_latest': 'medllama2:latest',
    'mistral-nemo_latest': 'mistral-nemo:latest',
    'mistral-small_24b-instruct-2501-q4_K_M': 'mistral-small:24b-instruct-2501-q4_K_M',
    'mistral-small_latest': 'mistral-small:latest',
    'mistral_7b': 'mistral:7b',
    'mistral_v0.2': 'mistral:v0.2',
    'mixtral_8x22b': 'mixtral:8x22b',
    'mixtral_8x7b': 'mixtral:8x7b',
    'nemotron-mini_4b': 'nemotron-mini:4b',
    'nemotron_latest': 'nemotron:latest',
    'nezahatkorkmaz_deepseek-v3_latest': 'nezahatkorkmaz/deepseek-v3:latest',
    'phi3_14b': 'phi3:14b',
    'phi3_3.8b': 'phi3:3.8b',
    'phi3_latest': 'phi3:latest',
    'qwen2.5-coder_latest': 'qwen2.5-coder:latest',
    'qwen2.5_72b': 'qwen2.5:72b',
    'qwen2.5_latest': 'qwen2.5:latest',
    'qwq_latest': 'qwq:latest',
    'reflection_70b': 'reflection:70b',
    'smollm2_latest': 'smollm2:latest',
    'x_llama3.2-vision_11b': 'x/llama3.2-vision:11b'
}


def calculate_pabak(y_true, y_pred):
    agreement = np.mean(y_true == y_pred)
    pabak = 2 * agreement - 1
    return pabak


def generate_uniqueid(row):
    """
    Generates a consistent row ID based on the normalized description.
    Combines normalization and hashing into a single function.
    """
    # Normalize while preserving some punctuation
    if 'Record' in row:
        normalized_description = re.sub(r'[^a-zA-Z0-9 \-():]', '', row['Record'])
    elif 'record' in row:
        normalized_description = re.sub(r'[^a-zA-Z0-9 \-():]', '', row['record'])
    else:
        raise ValueError("Row does not contain 'Record' or 'record' column.")
    # Normalize whitespace
    normalized_description = ' '.join(normalized_description.split())

    key_string = f"{normalized_description}"
    id_record = hashlib.sha256(key_string.encode()).hexdigest()[:20]
    return id_record


def calculate_performance_metrics(y_true, y_pred, total_records, analyzed_records):
    """
    Calculates performance metrics, adjusted for the number of unanalyzed records.

    Args:
        y_true: A list or array of true labels (0 or 1).
        y_pred: A list or array of predicted labels (0 or 1).
        total_records: The total number of records in the dataset.
        analyzed_recorcalculate_performance_metricsds: The number of records that were successfully analyzed.

    Returns:
        A dictionary containing the calculated metrics.
    """

    print("\n--- Metrics Calculation ---")  # Add print statement to indicate start of metric calculation
    print(f"True labels distribution: {np.bincount(y_true)}")  # Print distribution of true labels
    print(f"Predicted labels distribution: {np.bincount(y_pred)}")  # Print distribution of predicted labels

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred,
                                zero_division=0)  # Keep zero_division=0 to handle the warning explicitly
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    pabak = calculate_pabak(y_true, y_pred)

    # Calculate the percentage of data that was analyzed
    analysis_coverage = analyzed_records / total_records if total_records > 0 else 0

    # Calculate adjustment factor
    if analyzed_records < total_records:
        adjustment_factor = analyzed_records / total_records
    else:
        adjustment_factor = 1.0  # No adjustment needed if all records were analyzed

    # If the adjustment factor is very small, metrics might not be meaningful
    if adjustment_factor < 0.5:  # You can adjust this threshold as needed
        print(f"Warning: Only {adjustment_factor:.2%} of records were analyzed. Metrics might be unreliable.")

    # Apply adjustment factor only if needed
    adjusted_accuracy = accuracy * adjustment_factor
    adjusted_precision = precision * adjustment_factor
    adjusted_recall = recall * adjustment_factor
    adjusted_f1 = f1 * adjustment_factor
    adjusted_mcc = mcc * adjustment_factor
    adjusted_kappa = kappa * adjustment_factor
    adjusted_pabak = pabak * adjustment_factor

    return {
        'TP': np.sum((y_true == 1) & (y_pred == 1)),
        'TN': np.sum((y_true == 0) & (y_pred == 0)),
        'FP': np.sum((y_true == 0) & (y_pred == 1)),
        'FN': np.sum((y_true == 1) & (y_pred == 0)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "pabak": pabak,
        "adjusted_accuracy": adjusted_accuracy,
        "adjusted_precision": adjusted_precision,
        "adjusted_recall": adjusted_recall,
        "adjusted_f1_score": adjusted_f1,
        "adjusted_mcc": adjusted_mcc,
        "adjusted_cohen_kappa": adjusted_kappa,
        "adjusted_pabak": adjusted_pabak,
        "analysis_coverage": analysis_coverage,
        "analyzed_records": analyzed_records,
        "total_records": total_records
    }


def process_review(review, screening_type, criteria_threshold=None):
    """
        Processes a review and calculates performance metrics for a given screening type.

        Args:
            review (dict): A dictionary containing review configurations.
            screening_type (str): The type of screening ('screening1' or 'screening2').
            criteria_threshold (int, optional): The minimum number of criteria that must be True
                                                for a record to be classified as 'predicted_screening' True.
                                                If None, all criteria must be True (default behavior).

        Returns:
            tuple: A tuple containing:
                - results_dfs (list): A list of DataFrames containing detailed results for each model.
                - performance_results (pd.DataFrame): A DataFrame summarizing performance metrics for each model.

            review = rw_3_workdir_scr1
            screening_type = 'screening2'
                        screening_type = 'screening1'

            criteria_threshold=None
        """

    results_dfs = {}
    results_performance = []

    working_directory = review['directory']
    results_directory = review['results_directory']
    og_df = pd.read_pickle(f"{working_directory}preprocessed_articles_filtered.pkl")
    og_df.reset_index(drop=True, inplace=True)
    if 'record' in og_df.columns:
        og_df = og_df.rename(columns={'record': 'Record'})

    og_df['uniqueid'] = og_df.apply(generate_uniqueid, axis=1)
    og_df['rec-number'] = og_df['uniqueid']

    if screening_type == 'screening2':
        og_df = og_df[og_df.screening1 == True].copy()
        og_df = og_df[og_df['pdf_name'] != 'NO PAPER AVAILABLE.pdf']

    # First, collect all the data
    total_records = len(og_df)
    print(f"Original DataFrame shape: {og_df.shape}")

    for model in tqdm(review['model_list']):
        # model = 'llama3.3_70b'
        # model = 'gemini-2.0-flash-lite'
        print(f"\nProcessing model: {model}")

        try:
            results_model = f"{results_directory}results_{model}/results_{screening_type}.pkl"
            with open(results_model, 'rb') as file:
                detailed_df = pickle.load(file)

            # Load both predicted criteria and missing records
            predicted_criteria_path = f"{results_directory}results_{model}/{screening_type}/{screening_type}_predicted_criteria.pkl"
            missing_records_path = f"{results_directory}results_{model}/{screening_type}/{screening_type}_missing_records.pkl"

            with open(predicted_criteria_path, 'rb') as file:
                predicted_criteria = pickle.load(file)

            missing_records = set()
            if os.path.exists(missing_records_path):
                with open(missing_records_path, 'rb') as file:
                    missing_records = pickle.load(file)

            print(f"Number of successfully analyzed records: {len(predicted_criteria)}")
            print(f"Number of records that couldn't be analyzed: {len(missing_records)}")
            results_dfs[model] = detailed_df

            y_true = detailed_df[screening_type]
            y_pred = detailed_df[f'predicted_{screening_type}']
            analyzed_records = len(predicted_criteria)

            resdict = calculate_performance_metrics(y_true, y_pred, total_records, analyzed_records)
            resdict['model'] = model
            resdict['total_records'] = total_records
            resdict['analyzed_records'] = analyzed_records
            resdict['missing_records'] = len(missing_records)
            resdict['screening_type'] = screening_type
            resdict['model_name'] = model_name_corrections.get(model, model)
            resdict['reviewname'] = review['name']
            resdict['criteriathr'] = criteria_threshold

            results_performance.append(resdict)

            if screening_type == 'screening1':
                dataset_path = f"{results_directory}results_{model}/results_screening1.pkl"
            elif screening_type == 'screening2':
                dataset_path = f"{results_directory}results_{model}/results_screening2.pkl"
            detailed_df.to_pickle(dataset_path)

        except Exception as e:
            print(f"Error processing model {model}: {str(e)}")
            continue
    performance_results = pd.DataFrame.from_records(results_performance)

    return results_dfs, performance_results


def compute_criteria_pearson_correlation(predcrit_dict, screening_type, config):
    """
    Computes the Pearson correlation for each criterion (from config["columns_needed_as_true"])
    with respect to the screening target (screening1 or screening2).

    Parameters:
        detailed_df (pd.DataFrame): DataFrame containing the detailed results. It must include
                                    the screening target column and criteria columns.
        screening_type (str): Either 'screening1' or 'screening2'.
        config (dict): The configuration dictionary (e.g., rw_1_workdir_scr1 or rw_1_workdir_scr2).

    Returns:
        pd.DataFrame: DataFrame with columns 'Criterion', 'Screening', and 'Correlation'.

        detailed_df =  results1['mistral-nemo_latest']
        config = rw_1_workdir_scr1
        predcrit_dict =  results1
        screening_type = 'screening1'
    """

    results = []
    for key in predcrit_dict.keys():
        print("model name: ", key)

        detailed_df = predcrit_dict[key]
        target = screening_type
        # Determine the suffix based on screening type.
        suffix = '_scr1' if screening_type == 'screening1' else '_scr2'
        criteria = config["columns_needed_as_true"]
        for crit in criteria:
            col_name = crit.replace(suffix, '')
            if crit in detailed_df.columns:
                # Compute Pearson correlation (True is 1, False is 0)
                corr_val = detailed_df[crit].corr(detailed_df[target])
                results.append({
                    'Criterion': col_name,
                    'Screening': screening_type,
                    'Correlation': corr_val,
                    'model': key,
                })
    return pd.DataFrame(results)

def plot_performance_metrics_grouped(
    df_results, metrics, model_order, save_path=None, dpi=300,
    custom_colors=None, plot_title=None, add_value_labels=True,
    fig_width=12, fig_height=8
):
    """
    Creates performance metric plots for comparing model performance
    across different screenings and reviews (grouped bar plot).
    Matches the style of the provided example with consistent colors and spacing.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set up enhanced plotting style
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 10,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'axes.facecolor': '#ffffff',
        'figure.facecolor': '#ffffff',
    })
    
    # Create figure with custom size
    fig, axes = plt.subplots(
        len(metrics), 1, 
        figsize=(fig_width, fig_height), 
        sharex=True,
        constrained_layout=True,
    )
    
    if len(metrics) == 1:
        axes = [axes]  # Make axes iterable if there's only one subplot
    
    # Define color scheme to match the example image
    default_colors = {
        ('I', 'screening1'): '#B5C7E9',  # Light blue
        ('I', 'screening2'): '#5B6A84',  # Dark blue
        ('II', 'screening1'): '#FBD78B',  # Light orange/yellow
        ('II', 'screening2'): '#E9A064',  # Darker orange
    }
    
    # Use custom colors if provided, otherwise use default palette
    colors_to_use = custom_colors if custom_colors else default_colors
    
    # For each metric, create a subplot
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Group by model, reviewname, and screening_type to get the mean metric value
        grouped = df_results.groupby(['model', 'reviewname', 'screening_type'])[metric].mean().reset_index()
        
        # Filter to only include models in the specified order
        grouped = grouped[grouped['model'].isin(model_order)]
        
        # Sort by the model order
        grouped['model_order'] = grouped['model'].map({m: i for i, m in enumerate(model_order)})
        grouped = grouped.sort_values('model_order')
        
        # Set up the bar positions
        bar_width = 0.18  # Match the example spacing
        models = grouped['model'].unique()
        n_models = len(models)
        
        # Get unique review/screening combinations
        combinations = grouped[['reviewname', 'screening_type']].drop_duplicates()
        combinations = combinations.sort_values(['reviewname', 'screening_type'])
        n_combinations = len(combinations)
        
        # Calculate positions for each group of bars
        positions = np.arange(n_models)
        
        # Plot each review/screening combination as a separate bar
        for j, (_, row) in enumerate(combinations.iterrows()):
            review = row['reviewname']
            screening = row['screening_type']
            
            # Format label with "Review" prefix and capital S in "Screening"
            label = f"Review {review} - Screening {screening[-1]}"
            
            # Filter data for this review/screening
            subset = grouped[(grouped['reviewname'] == review) & 
                             (grouped['screening_type'] == screening)]
            
            # Calculate bar position
            offset = (j - n_combinations/2 + 0.5) * bar_width
            bar_pos = positions + offset
            
            # Get color for this review/screening
            color = colors_to_use.get((review, screening), '#CCCCCC')  # Default to gray if not found
            
            # Plot the bars with white borders and straight edges
            bars = ax.bar(
                bar_pos, 
                subset[metric], 
                width=bar_width,
                label=label,
                color=color,
                edgecolor='white',  # White edge color
                linewidth=3,        # Thicker border
                alpha=1.0,          # Full opacity
                zorder=3
            )
            
            # Add value labels if requested
            if add_value_labels:
                for bar in bars:
                    height = bar.get_height()
                    value_text = f"{height:.2f}"
                        
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.01,
                        value_text,
                        ha='center', 
                        va='bottom',
                        fontsize=9,
                        fontweight='normal', 
                        color='black',
                        zorder=4
                    )
        
        # Set the labels with improved styling
        metric_name = metric.replace('_', ' ').title()
        # Make MCC all caps
        metric_name = metric_name.replace('Mcc', 'MCC')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        
        # Set the x-tick positions and labels
        ax.set_xticks(positions)
        
        try:
            # Try to use model_name_corrections if it exists
            ax.set_xticklabels([model_name_corrections.get(m, m) for m in models], 
                       rotation=45, ha='right', fontsize=10, fontweight='bold')
        except NameError:
            # If model_name_corrections is not defined
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10, fontweight='bold')
        
        # Add horizontal grid lines only (to match example)
        ax.grid(True, axis='y', linestyle='-', alpha=0.15, color='#999999', zorder=0)
        ax.grid(False, axis='x')  # Remove vertical grid lines
        
        # Set y-axis limits with padding
        ax.set_ylim(0, min(0.8, ax.get_ylim()[1] * 1.1))  # Cap at 0.8 to match example
        
        # Remove spines for a cleaner look
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        # Make remaining spines lighter
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#dddddd')
    
    # Add a single legend at the bottom of the figure (using handles from just one subplot to avoid duplication)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicates
    legend = fig.legend(
        by_label.values(), by_label.keys(),
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05),  # Position at bottom
        ncol=4,                      # Four items per row
        frameon=False,               # No frame
    )
    
    # Add main title if provided
    if plot_title:
        plot_title = plot_title.replace('Mcc', 'MCC')
        fig.suptitle(plot_title, fontsize=16, y=0.98, fontweight='bold')
    
    # Adjust layout for the legend and increase space between subplots
    plt.subplots_adjust(bottom=0.15, hspace=0.4)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig, axes


def plot_correlation_tiles(
    df_corr, criteria_mapping, model_name_corrections, 
    review_name=None, size_factor=3000, save_path=None,
    custom_cmap=None, add_annotations=True,
    fig_width=None, fig_height=None
):
    """
    Plots a bubble-tile plot showing correlation values for each model and criterion.
    """
    # Set up enhanced plotting style
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'axes.grid': False,
        'axes.facecolor': '#ffffff',
        'figure.facecolor': '#ffffff',
    })
    
    # Create a custom colormap if specified
    if isinstance(custom_cmap, str):
        cmap = plt.get_cmap(custom_cmap)
    elif custom_cmap is not None:
        cmap = custom_cmap
    else:
        # Default pastel colormap using the specified colors
        cmap = LinearSegmentedColormap.from_list(
            "pastel_diverging", 
            ["#E080A2", "#ffffff", "#6BAEE0"], 
            N=256
        )
    
    # Get unique screenings and models
    screenings = df_corr['Screening'].unique()
    
    # Compute overall model order (sorted by average correlation across screenings)
    model_order = (
        df_corr.groupby('model')['Correlation']
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    
    # Create figure with custom size if provided
    if fig_width and fig_height:
        fig_size = (fig_width, fig_height)
    else:
        fig_size = (6 * len(screenings), 0.5 * len(model_order) + 3)
    
    fig, axes = plt.subplots(
        1, len(screenings), 
        figsize=fig_size, 
        sharey=True,
        constrained_layout=True,
        facecolor='#ffffff'
    )
    
    if len(screenings) == 1:
        axes = [axes]  # Make axes iterable if there's only one subplot
    
    # Loop over each screening type
    for i, screening in enumerate(screenings):
        ax = axes[i]
        ax.set_facecolor('#ffffff')  # Light background
        
        # Subset and pivot the data
        sub_df = df_corr[df_corr['Screening'] == screening]
        pivot_df = sub_df.pivot_table(index='model', columns='Criterion', values='Correlation', aggfunc='mean')
        
        # Reorder rows by model_order and sort columns alphabetically
        pivot_df = pivot_df.reindex(model_order)
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
        
        # Get raw labels from the pivot table
        x_labels = pivot_df.columns.tolist()
        y_labels = pivot_df.index.tolist()
        
        # Set up the axes
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9, fontweight='bold')
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels([model_name_corrections.get(m, m) for m in y_labels], fontsize=9)
        
        # Remove spines for a cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Draw white gridlines between cells
        for x in np.arange(-0.5, len(x_labels), 1):
            ax.axvline(x, color='white', linewidth=1.5, zorder=1)
        for y in np.arange(-0.5, len(y_labels), 1):
            ax.axhline(y, color='white', linewidth=1.5, zorder=1)
        
        # Plot each cell as a bubble
        for yi, model in enumerate(y_labels):
            for xi, crit in enumerate(x_labels):
                corr = pivot_df.loc[model, crit]
                if pd.isna(corr):
                    continue
                
                # Scale marker size by correlation strength
                marker_size = abs(corr) * size_factor
                
                # Map correlation from [-1,1] to [0,1] for colormap
                color_val = (corr + 1) / 2
                color = cmap(color_val)
                
                # Plot the bubble with a subtle shadow effect
                ax.scatter(
                    xi, yi, 
                    s=marker_size, 
                    color=color, 
                    edgecolors='#ffffff', 
                    linewidth=0.5, 
                    alpha=0.85, 
                    zorder=2
                )
                
                # Add text annotation if requested
                if add_annotations:
                    # Choose text color based on bubble color brightness
                    text_color = 'black' if color_val > 0.4 else 'white'
                    
                    ax.text(
                        xi, yi, 
                        f"{corr:.2f}", 
                        ha='center', 
                        va='center', 
                        fontsize=9,  # Minimum font size
                        fontweight='normal',
                        color=text_color, 
                        zorder=3
                    )
        
        # Set title for each screening panel with improved styling
        screening_titles = {
            'screening1': 'Title & Abstract Screening',
            'screening2': 'Full-Text Screening'
        }
        ax.set_title(
            screening_titles.get(screening, screening), 
            fontsize=14, 
            fontweight='bold', 
            pad=15
        )
        
        # Set axis limits
        ax.set_xlim(-0.5, len(x_labels) - 0.5)
        ax.set_ylim(len(y_labels) - 0.5, -0.5)  # Invert y-axis
    
    # Add a main title if provided
    if review_name:
        # Replace MCC with all caps if present
        review_name = review_name.replace('Mcc', 'MCC')
        fig.suptitle(
            review_name, 
            fontsize=16, 
            fontweight='bold', 
            y=1.05
        )
        

    
    # # Add a colorbar legend at the bottom
    # cbar_ax = fig.add_axes([0.3, -0.05, 0.4, 0.02])  # [left, bottom, width, height]
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    # sm.set_array([])
    # cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    # cbar.set_label('Correlation Strength', fontsize=9, fontweight='bold')
    
    # Adjust layout to make room for the legend at the bottom
    plt.subplots_adjust(bottom=0.15)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Plot saved to {save_path}")
    
    return fig, axes


def process_review_repredicting(review, screening_type, criteria_threshold=None):
    """
        Processes a review and calculates performance metrics for a given screening type.

        Args:
            review (dict): A dictionary containing review configurations.
            screening_type (str): The type of screening ('screening1' or 'screening2').
            criteria_threshold (int, optional): The minimum number of criteria that must be True
                                                for a record to be classified as 'predicted_screening' True.
                                                If None, all criteria must be True (default behavior).

        Returns:
            tuple: A tuple containing:
                - results_dfs (list): A list of DataFrames containing detailed results for each model.
                - performance_results (pd.DataFrame): A DataFrame summarizing performance metrics for each model.

            review = rw_1_workdir_scr1
            screening_type = 'screening1'
            criteria_threshold=None
        """

    results_dfs = {}
    results_performance = []

    working_directory = review['directory']
    results_directory = review['results_directory']
    og_df = pd.read_pickle(f"{working_directory}preprocessed_articles_filtered.pkl")
    og_df.reset_index(drop=True, inplace=True)
    if 'record' in og_df.columns:
        og_df = og_df.rename(columns={'record': 'Record'})

    og_df['uniqueid'] = og_df.apply(generate_uniqueid, axis=1)
    og_df['rec-number'] = og_df['uniqueid']

    if screening_type == 'screening2':
        og_df = og_df[og_df.screening1 == True].copy()



    # First, collect all the data
    total_records = len(og_df)
    print(f"Original DataFrame shape: {og_df.shape}")

    for model in tqdm(review['model_list']):
        # model = 'llama3.3_70b'
        # model = 'gemini-2.0-flash-lite'
        # model = '
        print(f"\nProcessing model: {model}")

        try:
            results_model = f"{results_directory}results_{model}/results_{screening_type}.pkl"
            with open(results_model, 'rb') as file:
                results_model = pickle.load(file)

            # Load both predicted criteria and missing records
            predicted_criteria_path = f"{results_directory}results_{model}/{screening_type}/{screening_type}_predicted_criteria.pkl"
            missing_records_path = f"{results_directory}results_{model}/{screening_type}/{screening_type}_missing_records.pkl"

            with open(predicted_criteria_path, 'rb') as file:
                predicted_criteria = pickle.load(file)

            missing_records = set()
            if os.path.exists(missing_records_path):
                with open(missing_records_path, 'rb') as file:
                    missing_records = pickle.load(file)

            print(f"Number of successfully analyzed records: {len(predicted_criteria)}")
            print(f"Number of records that couldn't be analyzed: {len(missing_records)}")

            # Transform the criteria data with improved label handling and include reasons
            transformed_data = {}
            reasons_data = {}  # Store reasons separately

            for index, item in predicted_criteria.items():

                transformed_data[str(index)] = {}
                reasons_data[str(index)] = {}

                for criterion, entry in item.items():
                    if isinstance(entry, dict):
                        label = entry.get('label', False)
                        reason = entry.get('reason', '')  # Get the reason if available
                    else:
                        label = entry
                        reason = ''

                    if isinstance(label, bool):
                        transformed_data[str(index)][criterion] = label
                    elif isinstance(label, str):
                        label_lower = label.lower().strip()
                        if label_lower == 'true' or label_lower == 't' or label_lower == 'yes':
                            transformed_data[str(index)][criterion] = True
                        elif label_lower == 'false' or label_lower == 'f' or label_lower == 'no':
                            transformed_data[str(index)][criterion] = False
                        else:
                            transformed_data[str(index)][criterion] = False
                    else:
                        transformed_data[str(index)][criterion] = False

                    reasons_data[str(index)][f"{criterion}_reason"] = reason

            if not transformed_data:
                print(f"WARNING: No valid records found for model {model}")

            # Create DataFrame from criteria data
            criteria_df = pd.DataFrame.from_dict(transformed_data, orient='index')
            reasons_df = pd.DataFrame.from_dict(reasons_data, orient='index')

            if len(criteria_df) == 0:
                print(f"WARNING: Empty criteria DataFrame for model {model}")

            # Ensure all values are boolean
            for col in criteria_df.columns:
                criteria_df[col] = criteria_df[col].astype(bool)

            # Append '_scr1' to all column names in criteria_df
            if screening_type == 'screening1':
                criteria_df.columns = [col + '_scr1' for col in criteria_df.columns]
            elif screening_type == 'screening2':
                criteria_df.columns = [col + '_scr2' for col in criteria_df.columns]

            # Create the new 'predicted_screening' column based on criteria_threshold
            if criteria_threshold is None:
                # Default behavior: all criteria must be True
                criteria_df[f'predicted_{screening_type}'] = criteria_df.apply(lambda row: all(row), axis=1)
            else:
                # Custom threshold: at least criteria_threshold criteria must be True
                criteria_df[f'predicted_{screening_type}'] = criteria_df.apply(
                    lambda row: sum(row) >= criteria_threshold, axis=1)

            # Reset the index and rename it to 'rec-number' to match og_df
            criteria_df = criteria_df.reset_index().rename(columns={'index': 'rec-number'})
            reasons_df = reasons_df.reset_index().rename(columns={'index': 'rec-number'})

            detailed_df = og_df[['rec-number', 'Record', 'screening1', 'screening2']].merge(
                criteria_df,
                on='rec-number',
                how='right'
            )

            # detailed_df = results_model

            results_dfs[model] = detailed_df

            y_true = detailed_df[screening_type].astype(int)
            y_pred = detailed_df[f'predicted_{screening_type}'].astype(int)
            analyzed_records = len(predicted_criteria)
            resdict = calculate_performance_metrics(y_true, y_pred, total_records, analyzed_records)
            resdict['model'] = model
            resdict['total_records'] = total_records
            resdict['analyzed_records'] = analyzed_records
            resdict['missing_records'] = len(missing_records)
            resdict['screening_type'] = screening_type
            resdict['model_name'] = model_name_corrections.get(model, model)
            resdict['reviewname'] = review['name']
            resdict['criteriathr'] = criteria_threshold

            results_performance.append(resdict)

            if screening_type == 'screening1':
                dataset_path = f"{results_directory}results_{model}/results_screening1.pkl"
            elif screening_type == 'screening2':
                dataset_path = f"{results_directory}results_{model}/results_screening2.pkl"
            detailed_df.to_pickle(dataset_path)

        except Exception as e:
            print(f"Error processing model {model}: {str(e)}")
            continue
    performance_results = pd.DataFrame.from_records(results_performance)

    return results_dfs, performance_results


# First, find the intersection of models across all reviews
def fix_model_analysis_workflow(performance_results_scr1, performance_results_scr2):
    # Get unique review names
    reviews_scr1 = performance_results_scr1['reviewname'].unique()
    reviews_scr2 = performance_results_scr2['reviewname'].unique()
    all_reviews = set(reviews_scr1) | set(reviews_scr2)

    # Find models that appear in all reviews for both screening types
    models_per_review = {}

    # Check screening1
    for review in reviews_scr1:
        models_in_review = set(performance_results_scr1[performance_results_scr1['reviewname'] == review]['model'])
        models_per_review[f"scr1_{review}"] = models_in_review

    # Check screening2
    for review in reviews_scr2:
        models_in_review = set(performance_results_scr2[performance_results_scr2['reviewname'] == review]['model'])
        models_per_review[f"scr2_{review}"] = models_in_review

    # Find intersection of models across all reviews and screening types
    common_models = set.intersection(*models_per_review.values())

    print(f"Number of models common across all reviews: {len(common_models)}")

    if len(common_models) == 0:
        print("ERROR: No common models found across all reviews!")
        # Fallback: find models common within each screening type
        common_scr1 = set.intersection(*[models_per_review[f"scr1_{r}"] for r in reviews_scr1])
        common_scr2 = set.intersection(*[models_per_review[f"scr2_{r}"] for r in reviews_scr2])
        print(f"Models common across screening1: {len(common_scr1)}")
        print(f"Models common across screening2: {len(common_scr2)}")
        # Use the intersection of both screening types
        common_models = common_scr1.intersection(common_scr2)
        print(f"Using {len(common_models)} models common across both screening types")

    # Filter results to only include common models
    filtered_scr1 = performance_results_scr1[performance_results_scr1['model'].isin(common_models)]
    filtered_scr2 = performance_results_scr2[performance_results_scr2['model'].isin(common_models)]

    # Combine both screening results
    results_df_all = pd.concat([filtered_scr1, filtered_scr2])

    # Calculate model rankings based on the filtered data
    metrics_to_avg = ['adjusted_mcc']
    df_avg = results_df_all.groupby(['model', 'screening_type'])[metrics_to_avg].mean().groupby('model').mean()

    # Rank models by average adjusted_mcc
    ranking_df = df_avg.reset_index()
    ranking_df.columns = ['model', 'average_adjusted_mcc']
    ranking_df['rank'] = ranking_df['average_adjusted_mcc'].rank(ascending=False, method='min').astype(int)
    ranking_df = ranking_df.sort_values(by='average_adjusted_mcc', ascending=False)

    # Get final model order based on ranking
    model_order = ranking_df['model'].tolist()

    # You could optionally limit to top K models
    # top_k = model_order[:15]
    # results_df_all = results_df_all[results_df_all['model'].isin(top_k)]

    return results_df_all, model_order, ranking_df


def process_review_with_clustering(review, screening_type, variance_threshold=0.15):
    """
    Enhanced review processing using KMeans clustering instead of direct prediction.
    """
    results_dfs = {}
    results_performance = []

    # Load original data and prepare it as you already do
    working_directory = review['directory']
    results_directory = review['results_directory']
    og_df = pd.read_pickle(f"{working_directory}preprocessed_articles_filtered.pkl")
    og_df.reset_index(drop=True, inplace=True)
    if 'record' in og_df.columns:
        og_df = og_df.rename(columns={'record': 'Record'})

    og_df['uniqueid'] = og_df.apply(generate_uniqueid, axis=1)
    og_df['rec-number'] = og_df['uniqueid']

    if screening_type == 'screening2':
        og_df = og_df[og_df.screening1 == True].copy()

    total_records = len(og_df)
    print(f"Original DataFrame shape: {og_df.shape}")

    for model in tqdm(review['model_list']):
        print(f"\nProcessing model: {model}")

        try:
            # Load model results as you already do
            results_model = f"{results_directory}results_{model}/results_{screening_type}.pkl"
            with open(results_model, 'rb') as file:
                results_model = pickle.load(file)

            # Load criteria predictions
            predicted_criteria_path = f"{results_directory}results_{model}/{screening_type}/{screening_type}_predicted_criteria.pkl"
            missing_records_path = f"{results_directory}results_{model}/{screening_type}/{screening_type}_missing_records.pkl"

            with open(predicted_criteria_path, 'rb') as file:
                predicted_criteria = pickle.load(file)

            missing_records = set()
            if os.path.exists(missing_records_path):
                with open(missing_records_path, 'rb') as file:
                    missing_records = pickle.load(file)

            # Transform the criteria data into a boolean matrix
            transformed_data = {}
            for index, item in predicted_criteria.items():
                transformed_data[str(index)] = {}
                for criterion, entry in item.items():
                    if isinstance(entry, dict):
                        label = entry.get('label', False)
                    else:
                        label = entry

                    if isinstance(label, bool):
                        transformed_data[str(index)][criterion] = label
                    elif isinstance(label, str):
                        label_lower = label.lower().strip()
                        transformed_data[str(index)][criterion] = label_lower in ['true', 't', 'yes']
                    else:
                        transformed_data[str(index)][criterion] = False

            # Create DataFrame and boolean matrix
            criteria_df = pd.DataFrame.from_dict(transformed_data, orient='index')
            boolean_matrix = criteria_df.values
            feature_names = criteria_df.columns.tolist()

            # Apply clustering
            team_labels, selected_features = identify_teams(boolean_matrix, variance_threshold=variance_threshold)

            # Add the cluster labels to the criteria DataFrame
            criteria_df[f'cluster_{screening_type}'] = team_labels

            # Create the new screening prediction based on clustering
            criteria_df[f'predicted_{screening_type}'] = team_labels

            # Analyze which features were most important for the clustering
            analysis = analyze_teams(boolean_matrix, team_labels, feature_names)
            print(f"Top discriminative features: {[f[0] for f in analysis['top_features'][:3]]}")
            print(f"Team sizes: {analysis['winning_team_size']} winning, {analysis['losing_team_size']} losing")

            # Append '_scr1' or '_scr2' to column names
            suffix = '_scr1' if screening_type == 'screening1' else '_scr2'
            rename_dict = {col: col + suffix for col in criteria_df.columns if col not in
                           [f'cluster_{screening_type}', f'predicted_{screening_type}']}
            criteria_df = criteria_df.rename(columns=rename_dict)

            # Prepare for merging
            criteria_df = criteria_df.reset_index().rename(columns={'index': 'rec-number'})

            # Merge with original data
            detailed_df = og_df[['rec-number', 'Record', 'screening1', 'screening2']].merge(
                criteria_df, on='rec-number', how='right'
            )

            results_dfs[model] = detailed_df

            # Calculate performance metrics
            y_true = detailed_df[screening_type].astype(int)
            y_pred = detailed_df[f'predicted_{screening_type}'].astype(int)
            analyzed_records = len(predicted_criteria)

            resdict = calculate_performance_metrics(y_true, y_pred, total_records, analyzed_records)
            resdict['model'] = model
            resdict['total_records'] = total_records
            resdict['analyzed_records'] = analyzed_records
            resdict['missing_records'] = len(missing_records)
            resdict['screening_type'] = screening_type
            resdict['model_name'] = model_name_corrections.get(model, model)
            resdict['reviewname'] = review['name']
            resdict['method'] = 'clustering'

            results_performance.append(resdict)

            # Save results
            dataset_path = f"{results_directory}results_{model}/results_{screening_type}_clustered.pkl"
            detailed_df.to_pickle(dataset_path)

        except Exception as e:
            print(f"Error processing model {model}: {str(e)}")
            continue

    performance_results = pd.DataFrame.from_records(results_performance)

    return results_dfs, performance_results



def identify_teams(boolean_matrix, variance_threshold=0.15):
    """
    Identify winning and losing teams from a boolean feature matrix.

    Parameters:
    -----------
    boolean_matrix : numpy.ndarray
        A matrix where rows are instances and columns are boolean features (0/1)
    variance_threshold : float, default=0.15
        Threshold for feature selection (0-0.25 for boolean data)

    Returns:
    --------
    team_labels : numpy.ndarray
        Array of labels (1 for winning team, 0 for losing team)
    selected_features : list
        Indices of discriminative features used for clustering
    """
    # Feature selection - identify discriminative features
    feature_variances = np.var(boolean_matrix, axis=0)
    selected_features = np.where(feature_variances >= variance_threshold)[0]

    # If no features meet threshold, use top 3 features by variance
    if len(selected_features) == 0:
        selected_features = np.argsort(-feature_variances)[:3]

    # Clustering using selected features
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(boolean_matrix[:, selected_features])

    # Identify winning team (more true values across all features)
    team0_avg = np.mean(boolean_matrix[cluster_labels == 0])
    team1_avg = np.mean(boolean_matrix[cluster_labels == 1])
    winning_team = 0 if team0_avg > team1_avg else 1

    # Return binary labels (1=winning, 0=losing) and selected features
    return (cluster_labels == winning_team).astype(int), selected_features


def analyze_teams(boolean_matrix, team_labels, feature_names=None):
    """
    Analyze the differences between winning and losing teams.

    Parameters:
    -----------
    boolean_matrix : numpy.ndarray
        A matrix where rows are instances and columns are boolean features (0/1)
    team_labels : numpy.ndarray
        Array of labels (1 for winning team, 0 for losing team)
    feature_names : list, optional
        Names of features (uses indices if None)

    Returns:
    --------
    dict
        Dictionary with feature importance information
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(boolean_matrix.shape[1])]

    # Calculate feature means for each team
    winning_means = np.mean(boolean_matrix[team_labels == 1], axis=0)
    losing_means = np.mean(boolean_matrix[team_labels == 0], axis=0)

    # Calculate differences and sort by importance
    differences = winning_means - losing_means
    importance_order = np.argsort(-np.abs(differences))

    return {
        "top_features": [(feature_names[i], differences[i]) for i in importance_order],
        "winning_team_size": np.sum(team_labels == 1),
        "losing_team_size": np.sum(team_labels == 0),
        "winning_team_avg_true": np.mean(boolean_matrix[team_labels == 1]),
        "losing_team_avg_true": np.mean(boolean_matrix[team_labels == 0])
    }


def performance_by_screening(output_df, screening_type, infodict):

    y_true = output_df[screening_type].astype(int)
    y_pred = output_df[f'predicted_{screening_type}'].astype(int)
    total_records = infodict['total_records']
    analyzed_records = len(output_df)
    resdict = calculate_performance_metrics(y_true, y_pred, total_records, analyzed_records)

    resdict['total_records'] = total_records
    resdict['analyzed_records'] = analyzed_records
    resdict['missing_records'] = total_records - analyzed_records
    resdict['screening_type'] = screening_type
    resdict.update(infodict)
    return resdict
