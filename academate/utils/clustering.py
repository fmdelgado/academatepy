import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



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

def identify_teams_continuous(matrix, variance_percentile=75, scale_features=True):
    """Identify winning and losing teams from a continuous feature matrix (0-1 values)."""
    # Feature selection based on variance
    feature_variances = np.var(matrix, axis=0)
    variance_threshold = np.percentile(feature_variances, variance_percentile)
    selected_features = np.where(feature_variances >= variance_threshold)[0]

    # Ensure we have at least 3 features
    if len(selected_features) < 3:
        selected_features = np.argsort(-feature_variances)[:3]

    # Extract selected feature data
    X_selected = matrix[:, selected_features]

    # Standardize features if requested
    if scale_features:
        X_selected = StandardScaler().fit_transform(X_selected)

    # Clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_selected)

    # Identify winning team (higher average values across all features)
    team0_avg = np.mean(matrix[cluster_labels == 0])
    team1_avg = np.mean(matrix[cluster_labels == 1])
    winning_team = 0 if team0_avg > team1_avg else 1

    # Return binary labels (1=winning, 0=losing) and selected features
    return (cluster_labels == winning_team).astype(int), selected_features


def analyze_continuous_teams(matrix, team_labels, feature_names=None):
    """Analyze the differences between winning and losing teams with continuous features."""
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(matrix.shape[1])]

    # Calculate statistics for each team
    winning_features = matrix[team_labels == 1]
    losing_features = matrix[team_labels == 0]

    winning_means = np.mean(winning_features, axis=0)
    losing_means = np.mean(losing_features, axis=0)

    # Feature importance based on mean difference
    differences = winning_means - losing_means
    importance_order = np.argsort(-np.abs(differences))

    # Calculate effect size for each feature (Cohen's d)
    effect_sizes = []
    for i in range(matrix.shape[1]):
        win_vals = winning_features[:, i]
        lose_vals = losing_features[:, i]

        # Pooled standard deviation
        n1, n2 = len(win_vals), len(lose_vals)
        s1, s2 = np.std(win_vals, ddof=1), np.std(lose_vals, ddof=1)

        # Avoid division by zero
        if n1 > 0 and n2 > 0 and (s1 > 0 or s2 > 0):
            pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
            effect_size = (np.mean(win_vals) - np.mean(lose_vals)) / (pooled_std or 1)
        else:
            effect_size = 0

        effect_sizes.append(effect_size)

    return {
        "top_features": [(feature_names[i], differences[i], effect_sizes[i])
                         for i in importance_order],
        "winning_team_size": np.sum(team_labels == 1),
        "losing_team_size": np.sum(team_labels == 0),
        "winning_team_avg": np.mean(winning_features),
        "losing_team_avg": np.mean(losing_features)
    }