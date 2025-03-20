import os
import pandas as pd
import sys
sys.path.append('/Users/fernando/Documents/Research/academatepy')
from validation.analysis_functions import *
from validation.cross_rater import create_agreement_plots, create_side_by_side_kappa_plots
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set up enhanced plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Replace your current color palette with this pastel version
# Custom color palette for reviews and screenings with pastel tones
color_palette = {
    # Review I colors - pastel blue family
    ('I', 'screening1'): '#67a9cf',  # Soft pastel blue
    ('I', 'screening2'): '#02818a',  # Lighter pastel blue
    
    # Review II colors - pastel pink/rose family
    ('II', 'screening1'): '#f6eff7',  # Soft pastel pink
    ('II', 'screening2'): '#bdc9e1',  # Lighter pastel pink
    
    # Review III colors - pastel green family
    ('III', 'screening1'): '#d5f4e6',  # Soft pastel green
    ('III', 'screening2'): '#e8f8f5',  # Lighter pastel green
    
    # Review IV colors - pastel purple/lavender family
    ('IV', 'screening1'): '#e6e6fa',  # Soft pastel lavender
    ('IV', 'screening2'): '#f0f0ff',  # Lighter pastel lavender
}

model_list = [
    "gpt-3.5-turbo-0125",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",

    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",

    "llama3.3:70b",
    # "llama3:8b-instruct-fp16",
    # "llama3:latest",
    # "meditron:70b",
    # "medllama2:latest",
    # "mistral-nemo:latest",
    # "mistral-small:24b-instruct-2501-q4_K_M",
    # "mistral-small:latest",
    # "mistral:7b",
    # "mistral:v0.2",
    # "mixtral:8x22b",
    # "mixtral:8x7b",
    # "nemotron-mini:4b",
    # "nemotron:latest",
    # "nezahatkorkmaz/deepseek-v3:latest",
    # "phi3:14b",
    # "phi3:3.8b",
    # "phi3:latest",
    # "qwen2.5-coder:latest",
    # "qwen2.5:72b",
    # "qwen2.5:latest",
    # "qwq:latest",
    # "reflection:70b",
    # "smollm2:latest",
    # "x/llama3.2-vision:11b"
]

modelsname_infolder = [x.replace("/", "_").replace(":", "_") for x in model_list]  # replace special characters
folder2modelname = dict(zip(modelsname_infolder, model_list))
workdir = "/Users/fernando/Documents/Research/academatepy/validation/"
result_type = "results"

result_type_dir = f"{workdir}results/{result_type}"
os.makedirs(result_type_dir, exist_ok=True)
os.makedirs(f"{result_type_dir}/plots", exist_ok=True)


rw_1_workdir_scr1 = {'directory': f"{workdir}PICOS/",
                     'results_directory': f"{workdir}PICOS/{result_type}/",
                     "boolean_column_set": ['screening1', 'screening2', 'population_scr1', 'intervention_scr1',
                                            'physio_and_other_scr1', 'e_interventions_scr1', 'control_group_scr1',
                                            'outcome_scr1', 'study_type_scr1', 'predicted_screening1'],
                     "columns_needed_as_true": ['population_scr1', 'intervention_scr1', 'physio_and_other_scr1',
                                                'e_interventions_scr1', 'control_group_scr1', 'outcome_scr1',
                                                'study_type_scr1'],
                     "model_list": modelsname_infolder,
                     "description": "Physiotherapy",
                     "name": "I"}

rw_1_workdir_scr2 = {'directory': f"{workdir}PICOS/",
                     'results_directory': f"{workdir}PICOS/{result_type}/",
                     "boolean_column_set": ['screening1', 'screening2', 'population_scr2', 'intervention_scr2',
                                            'physio_and_other_scr2', 'e_interventions_scr2', 'control_group_scr2',
                                            'outcome_scr2', 'study_type_scr2', 'predicted_screening2'],
                     "columns_needed_as_true": ['population_scr2', 'intervention_scr2', 'physio_and_other_scr2',
                                                'e_interventions_scr2', 'control_group_scr2', 'outcome_scr2',
                                                'study_type_scr2'],
                     "model_list": modelsname_infolder,
                     "description": "Physiotherapy",
                     "name": "I"}

rw_2_workdir_scr1 = {'directory': f"{workdir}reproduction/",
                     'results_directory': f"{workdir}reproduction/{result_type}/",
                     "boolean_column_set": ['screening1', 'screening2',
                                            'Disease_scr1', 'Treatment_scr1',
                                            'Population_scr1', 'Intervention_scr1', 'Human_scr1',
                                            'Preclinical_Clinical_scr1', 'Outcome_scr1', 'Publicationtype_scr1',
                                            'predicted_screening1'],
                     "columns_needed_as_true": ['Population_scr1', 'Intervention_scr1', 'Human_scr1',
                                                'Preclinical_Clinical_scr1', 'Outcome_scr1', 'Publicationtype_scr1'],
                     "model_list": modelsname_infolder,
                     "description": "Endometrial disorders",
                     "name": "II"}

rw_2_workdir_scr2 = {'directory': f"{workdir}reproduction/",
                     'results_directory': f"{workdir}reproduction/{result_type}/",
                     "boolean_column_set": ['screening1', 'screening2',
                                            'Disease_scr2', 'Treatment_scr2',
                                            'Population_scr2', 'Intervention_scr2', 'Human_scr2',
                                            'Preclinical_Clinical_scr2', 'Outcome_scr2', 'Publicationtype_scr2',
                                            'predicted_screening1'],
                     "columns_needed_as_true": ['Population_scr2', 'Intervention_scr2', 'Human_scr2',
                                                'Preclinical_Clinical_scr2', 'Outcome_scr2', 'Publicationtype_scr2'],
                     "model_list": modelsname_infolder,
                     "description": "Endometrial disorders",
                     "name": "II"}


criteria_mapping = {
    'population': 'Population',
    'intervention': 'Intervention',
    'physio_and_other': 'Physiotherapy and another treatment',
    'e_interventions': 'E-interventions',
    'control_group': 'Control Group',
    'outcome': 'Outcome',
    'study_type': 'Study type',
    'Disease': 'Disease',
    'Treatment': 'Treatment',
    'Human': 'Human',
    'Genetic': 'Genetic',
    'Results': 'Results',
    'AI_functionality_description': 'AI functionality description',
    'Economic_evaluation': 'Economic Evaluation',
    'Quantitative_healthcare_outcomes': 'Quantitative Healthcare Outcomes',
    'Relevance_AI_Healthcare': 'Relevance to AI in Healthcare',
    'AI_application_description': 'AI Application Description',
    'Economic_outcome_details': 'Economic Outcome Details',
    "Population": 'Population',
    "Intervention": "Intervention",
    "Preclinical_Clinical": "Preclinical or Clinical",
    "Outcome": "Outcome",
    "Publicationtype": "Publication Type"
}

# Process each review
performance_results_scr1 = []
detailed_df_scr1 = {}
for review in [rw_1_workdir_scr1, rw_2_workdir_scr1]:
    print(f"\nProcessing review: {review['name']}")
    try:
        results1, performance_results = process_review_repredicting(review, screening_type='screening1')
        # print(f"Processed {len(results)} models for review {review['name']}")
        # print(f"Number of records in final DataFrame: {len(results[0])}")
        performance_results_scr1.append(performance_results)
        detailed_df_scr1[review['name']] = results1
    except Exception as e:
        print(f"--------\nError processing review {review['name']}: {str(e)}\n---------")
    print("=" * 50)
performance_results_scr1 = pd.concat(performance_results_scr1)

performance_results_scr2 = []
detailed_df_scr2 = {}
for review in [rw_1_workdir_scr2, rw_2_workdir_scr2]:
    try:
        results2, performance_results = process_review_repredicting(review, screening_type='screening2')
        print(f"Processed {len(results2)} models for review {review['name']}")
        performance_results_scr2.append(performance_results)
        detailed_df_scr2[review['name']] = results2
    except Exception as e:
        print(f"--------\nError processing review {review['name']}: {str(e)}\n---------")
    print("=" * 50)
performance_results_scr2 = pd.concat(performance_results_scr2)

metrics_to_plot = [
    # 'accuracy',
    # 'precision',
    # 'recall',
    # 'f1_score',
    # 'mcc',
    'pabak',
    'cohen_kappa',
    # 'adjusted_accuracy',
    # 'adjusted_precision',
    # 'adjusted_recall',
    # 'adjusted_f1_score',
    # 'adjusted_mcc',
    # 'adjusted_cohen_kappa',
    # 'adjusted_pabak'
]

# Then, use this function to generate properly filtered results and model order
results_df_all, model_order, ranking_df = fix_model_analysis_workflow(performance_results_scr1,
                                                                      performance_results_scr2)

metric_to_avg = 'adjusted_mcc'  # You set this for averaging
# First, identify all unique review/screening combinations
review_screening_combinations = []
for review in results_df_all['reviewname'].unique():
    for screening in results_df_all['screening_type'].unique():
        review_screening_combinations.append((review, screening))

# Find models that have data for ALL combinations
complete_models = []
for model in results_df_all['model'].unique():
    has_all_combinations = True
    for review, screening in review_screening_combinations:
        data = results_df_all[(results_df_all['model'] == model) &
                              (results_df_all['reviewname'] == review) &
                              (results_df_all['screening_type'] == screening)]
        if data.empty:
            has_all_combinations = False
            break

    if has_all_combinations:
        complete_models.append(model)

print(f"Models with data for all reviews and screenings: {len(complete_models)}")

# Calculate performance only for complete models
performance_by_model = {}
for model in complete_models:
    # First average by review
    review_scores = {}
    for review in results_df_all['reviewname'].unique():
        # For each review, average across screening types
        screening_scores = []
        for screening in results_df_all['screening_type'].unique():
            data = results_df_all[(results_df_all['model'] == model) &
                                  (results_df_all['reviewname'] == review) &
                                  (results_df_all['screening_type'] == screening)]
            if not data.empty:
                avg_kappa = data[metric_to_avg].mean()
                screening_scores.append(avg_kappa)

        # Average across screenings for this review
        review_scores[review] = sum(screening_scores) / len(screening_scores)

    # Then average across reviews
    performance_by_model[model] = sum(review_scores.values()) / len(review_scores)

# Create ranking DataFrame
ranking_df = pd.DataFrame({
    'model': list(performance_by_model.keys()),
    f'avg_{metric_to_avg}': list(performance_by_model.values())
})
ranking_df = ranking_df.sort_values(f'avg_{metric_to_avg}', ascending=False)
model_order = ranking_df['model'].tolist()
top_k = model_order[: 10]
results_df_all = results_df_all[results_df_all.model.isin(top_k)]

results_df_all.groupby(['screening_type', 'model_name'])['adjusted_mcc'].mean().reset_index(
    name='mean_adjusted_mcc').sort_values(by='mean_adjusted_mcc', ascending=False)
results_df_all.groupby(['screening_type', 'model_name'])['cohen_kappa'].mean().reset_index(
    name='mean_cohen_kappa').sort_values(by='mean_cohen_kappa', ascending=False)

# Update the plot calls with additional styling parameters
plot_performance_metrics_grouped(
    df_results=results_df_all,
    metrics=metrics_to_plot,
    model_order=top_k,
    save_path=f"{result_type_dir}/plots/performance_plots_kappa.png",
    custom_colors=color_palette,
    plot_title="Model Performance: Kappa Metrics",
    add_value_labels=True,
    fig_width=14,
    fig_height=10
)

metrics_to_plot = [
    'adjusted_f1_score',
    'adjusted_mcc',
]
plot_performance_metrics_grouped(
    df_results=results_df_all,
    metrics=metrics_to_plot,
    model_order=top_k,
    save_path=f"{result_type_dir}/plots/performance_plots_mcc.png",
    custom_colors=color_palette,
    plot_title="Model Performance: F1 and MCC Metrics",
    add_value_labels=True,
    fig_width=14,
    fig_height=8
)

metrics_to_plot = [
    'adjusted_mcc',
    'cohen_kappa',
    # 'adjusted_f1_score',
]
# Replace the color palette with the specified colors
color_palette = {
    ('I', 'screening1'): '#C9D6FF',  # Blue
    ('I', 'screening2'): '#5A69A4',  # Light teal
    ('II', 'screening1'): '#FFCF99',  # Green
    ('II', 'screening2'): '#FFAA56',  # Yellow/gold
}

# Then use this color palette in your plot calls
plot_performance_metrics_grouped(
    df_results=results_df_all,
    metrics=metrics_to_plot,
    model_order=top_k,
    save_path=f"{result_type_dir}/plots/performance_plots_3metrics.png",
    custom_colors=color_palette,
    plot_title=None,
    add_value_labels=True,
    fig_width=12,
    fig_height=7
)

results_df_all.to_csv("test.csv")


# For criteria correlation analysis, we need to filter the detailed results similarly
def filter_detailed_results(detailed_results, common_models):
    filtered_results = {}
    for model_name in common_models:
        if model_name in detailed_results:
            filtered_results[model_name] = detailed_results[model_name]
    return filtered_results


# Example usage for criteria correlation analysis
i = 1

for reviewname in detailed_df_scr1.keys():
    print(reviewname)
    # reviewname = 'I'

    # Filter detailed results to only include common models
    results1 = filter_detailed_results(detailed_df_scr1[reviewname], top_k)
    results2 = filter_detailed_results(detailed_df_scr2[reviewname], top_k)

    if reviewname == 'I':
        reviewdata = [rw_1_workdir_scr1, rw_1_workdir_scr2, rw_1_workdir_scr1['description']]
    elif reviewname == 'II':
        reviewdata = [rw_2_workdir_scr1, rw_2_workdir_scr2, rw_2_workdir_scr2['description']]

    # subset_dict = {}
    # for key in top_k:
    #     if key in results1:
    #         subset_dict[key] = results1[key]
    #
    # print(f"Original dictionary: {results1}")
    # print(f"Selected subset: {subset_dict}")
    results1 = {k: results1[k] for k in top_k if k in results1}
    pearson_scr1 = compute_criteria_pearson_correlation(results1, 'screening1', reviewdata[0])
    print("Pearson Correlations for Screening1:")
    print(pearson_scr1)

    # Compute Pearson correlations for screening2.
    results2 = {k: results2[k] for k in top_k if k in results2}
    pearson_scr2 = compute_criteria_pearson_correlation(results2, 'screening2', reviewdata[1])
    print("Pearson Correlations for Screening2:")
    print(pearson_scr2)

    # -----------------------------
    # Plotting the Pearson Correlations
    # -----------------------------
    df_corr = pd.concat([pearson_scr1, pearson_scr2])
    print("MODELS ANALYZED", len(df_corr.model.unique()))
    # Combine the correlation results from both screenings.
    # Call the plotting function.
    df_corr['Criterion'] = df_corr['Criterion'].map(criteria_mapping)
    df_corr['model'] = df_corr['model'].map(model_name_corrections)

    plot_correlation_tiles(df_corr, criteria_mapping, model_name_corrections,
                           review_name=f"Review {i}: {reviewdata[2]}",
                           size_factor=3000,
                           save_path=f"{result_type_dir}/plots/correlation_plot_{reviewname}.png")
    i += 1

# After your existing model processing code
reviews = ['I', 'II', 'III', 'IV']  # Update with your review names
# reviews = ['I', 'II']  # Update with your review names

# Create side-by-side kappa plots for each review
output_dir = f"{result_type_dir}/plots/side_by_side_kappa"
os.makedirs(output_dir, exist_ok=True)

# Combine results into the format needed for the new function
results_dict = {
    'screening1': detailed_df_scr1,  # This should contain {'I': df1, 'III': df3, ...}
    'screening2': detailed_df_scr2  # This should contain {'I': df1, 'III': df3, ...}
}

# Create the plots
create_side_by_side_kappa_plots(
    results_dict=results_dict,
    reviews=reviews,
    output_dir=output_dir,
    subset_models=top_k  # Your list of top models
)


# create_agreement_plots(
#     results_dict=detailed_df_scr1,
#     reviews=reviews,
#     screening_types=['screening1'],
#     output_dir=f"{result_type_dir}/plots/screening1",
#     subset_models=top_k,
#     custom_colors=color_palette,
#     fig_width=14,
#     fig_height=10,
#     add_value_labels=True
# )

# create_agreement_plots(
#     results_dict=detailed_df_scr2,
#     reviews=reviews,
#     screening_types=['screening2'],
#     output_dir=f"{result_type_dir}/plots/screening2",
#     subset_models=top_k,
#     custom_colors=color_palette,
#     fig_width=14,
#     fig_height=10,
#     add_value_labels=True
# )
