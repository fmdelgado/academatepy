# academate/utils/visualization.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan
from typing import List, Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def get_embeddings(texts: List[str], embedding_model, batch_size: int = 10):
    """
    Get embeddings for a list of texts.

    Args:
        texts: List of texts to embed
        embedding_model: Embedding model to use
        batch_size: Batch size for embedding

    Returns:
        np.ndarray: Array of embeddings
    """
    embeddings = []

    # Process in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            batch_embeddings = embedding_model.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error embedding batch {i // batch_size}: {e}")
            # Use zeros as fallback
            for _ in range(len(batch_texts)):
                embeddings.append([0.0] * 768)  # Typical embedding dimension

    return np.array(embeddings)


def cluster_articles(
        df: pd.DataFrame,
        embedding_model,
        content_column: str = "Record",
        min_cluster_size: int = 2,
        min_samples: int = 1,
        random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Cluster articles using HDBSCAN on embeddings.

    Args:
        df: DataFrame containing articles
        embedding_model: Model to create embeddings
        content_column: Column containing article text
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        random_state: Random state for reproducibility

    Returns:
        tuple: (DataFrame with clusters, 2D coordinates, raw embeddings)
    """
    # Get article texts
    texts = df[content_column].fillna("").tolist()

    # Get embeddings
    logger.info("Creating embeddings")
    embeddings = get_embeddings(texts, embedding_model)

    # Standardize
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Cluster
    logger.info("Clustering with HDBSCAN")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom',  # 'eom' (Excess of Mass) tends to find more clusters
        metric='euclidean',
        prediction_data=True
    )
    cluster_labels = clusterer.fit_predict(scaled_embeddings)

    # Add to dataframe
    result_df = df.copy()
    result_df['cluster'] = cluster_labels
    result_df['cluster_prob'] = clusterer.probabilities_

    # Generate 2D projection with t-SNE
    logger.info("Creating t-SNE projection")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(df) - 1))
    coords_2d = tsne.fit_transform(scaled_embeddings)

    return result_df, coords_2d, embeddings


def get_cluster_keywords(
        df: pd.DataFrame,
        content_column: str = "Record",
        n_keywords: int = 10
) -> Dict[int, List[str]]:
    """
    Extract keywords that characterize each cluster.

    Args:
        df: DataFrame with cluster column
        content_column: Column containing text
        n_keywords: Number of keywords per cluster

    Returns:
        dict: Mapping of cluster IDs to keyword lists
    """
    # Create TF-IDF vectorizer
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words='english',
        min_df=2,
        ngram_range=(1, 2)
    )

    # Fit vectorizer to all texts
    all_texts = df[content_column].fillna("").tolist()
    X = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Get keywords for each cluster
    keywords = {}
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:
            # Skip noise cluster
            continue

        # Get indices of articles in this cluster
        cluster_indices = df[df['cluster'] == cluster_id].index

        # If no articles in cluster, skip
        if len(cluster_indices) == 0:
            continue

        # Get mean TF-IDF scores for this cluster
        cluster_vectors = X[cluster_indices]
        cluster_mean = np.array(cluster_vectors.mean(axis=0))[0]

        # Get top keywords
        top_indices = np.argsort(cluster_mean)[-n_keywords:][::-1]
        cluster_keywords = [feature_names[i] for i in top_indices]

        keywords[cluster_id] = cluster_keywords

    return keywords


def name_clusters_with_llm(
        df: pd.DataFrame,
        keywords: Dict[int, List[str]],
        llm,
        content_column: str = "Record"
) -> Dict[int, str]:
    """
    Use an LLM to name clusters based on keywords and sample texts.

    Args:
        df: DataFrame with cluster column
        keywords: Dictionary mapping cluster IDs to keywords
        llm: Language model for naming
        content_column: Column containing article text

    Returns:
        dict: Mapping of cluster IDs to names
    """
    prompt_template = """
You are a research topic expert helping to name clusters of academic articles.

CLUSTER KEYWORDS: {keywords}

SAMPLE ARTICLES:
{samples}

Task: Based on the keywords and sample articles, provide a short, descriptive name for this research topic cluster. 
The name should be 2-5 words and capture the main theme of these articles.

CLUSTER NAME:
"""

    cluster_names = {}

    for cluster_id, cluster_keywords in keywords.items():
        # Get sample texts
        cluster_texts = df[df['cluster'] == cluster_id][content_column].sample(
            min(3, len(df[df['cluster'] == cluster_id]))
        ).tolist()

        # Format samples
        samples = "\n\n".join(
            f"ARTICLE {i + 1}:\n{text[:500]}..." for i, text in enumerate(cluster_texts)
        )

        # Format prompt
        prompt = prompt_template.format(
            keywords=", ".join(cluster_keywords),
            samples=samples
        )

        # Get name from LLM
        try:
            response = llm.invoke(prompt)
            if hasattr(response, 'content'):
                name = response.content.strip()
            else:
                name = str(response).strip()

            # Clean up name
            name = name.replace("CLUSTER NAME:", "").strip()
            name = name.replace('"', '').replace("'", "").strip()

            cluster_names[cluster_id] = name
        except Exception as e:
            logger.error(f"Error naming cluster {cluster_id}: {e}")
            cluster_names[cluster_id] = f"Cluster {cluster_id}"

    return cluster_names


def visualize_clusters_interactive(
        df,
        coords_2d,
        cluster_names=None,
        title="Article Clusters",
        color_by='cluster'
):
    """Fixed version that correctly handles text columns"""
    import plotly.express as px

    # Create new DataFrame for visualization
    viz_df = pd.DataFrame({
        'x': coords_2d[:, 0],
        'y': coords_2d[:, 1],
        'cluster': df['cluster'].values,
    })

    # Properly handle the text column
    if 'title' in df.columns:
        viz_df['text'] = df['title'].values
    elif 'Record' in df.columns:  # Your content column
        viz_df['text'] = df['Record'].str.slice(0, 100).values  # Truncate for display
    else:
        viz_df['text'] = [f"Article {i}" for i in range(len(df))]

    # Add color_by column if it's different from cluster
    if color_by != 'cluster' and color_by in df.columns:
        viz_df[color_by] = df[color_by].values

    # Use cluster names if provided
    if cluster_names and color_by == 'cluster':
        viz_df['cluster_name'] = viz_df['cluster'].map(
            lambda x: cluster_names.get(x, f"Cluster {x}" if x != -1 else "Noise")
        )
        color_column = 'cluster_name'
    else:
        color_column = color_by

    # Create hover text
    viz_df['hover_text'] = viz_df.apply(
        lambda row: f"Article: {row['text']}<br>Cluster: {row.get('cluster_name', row['cluster'])}",
        axis=1
    )

    # Create figure
    fig = px.scatter(
        viz_df,
        x='x',
        y='y',
        color=color_column,
        hover_data=['hover_text'],
        title=title,
        opacity=0.8
    )

    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        legend_title="Clusters",
        font=dict(size=12),
        xaxis_title="",
        yaxis_title="",
        xaxis_showticklabels=False,
        yaxis_showticklabels=False
    )

    # Add annotations for cluster centers
    if color_by == 'cluster':
        for cluster_id in viz_df['cluster'].unique():
            if cluster_id == -1:
                continue  # Skip noise

            # Get cluster center
            cluster_points = viz_df[viz_df['cluster'] == cluster_id]
            center_x = cluster_points['x'].mean()
            center_y = cluster_points['y'].mean()

            # Get cluster name
            if cluster_names and cluster_id in cluster_names:
                name = cluster_names[cluster_id]
            else:
                name = f"Cluster {cluster_id}"

            # Add annotation
            fig.add_annotation(
                x=center_x,
                y=center_y,
                text=name,
                showarrow=False,
                font=dict(size=14, color='black', family="Arial Black"),
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='black',
                borderwidth=1,
                borderpad=4
            )

    return fig


def improved_cluster_viz(df, coords_2d, cluster_names=None, title="Article Clusters"):
    """
    Create an improved cluster visualization that handles overlapping labels and noisy points better.

    Args:
        df: DataFrame with cluster column
        coords_2d: 2D coordinates from t-SNE
        cluster_names: Dictionary mapping cluster IDs to cluster names
        title: Title for the plot

    Returns:
        Plotly figure
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    # Create visualization dataframe
    viz_df = pd.DataFrame({
        'x': coords_2d[:, 0],
        'y': coords_2d[:, 1],
        'cluster': df['cluster'].values
    })

    # Add any available metadata for hover text
    if 'title' in df.columns:
        viz_df['title'] = df['title'].values

    if 'Record' in df.columns:
        # Get first 100 characters for hover
        viz_df['abstract'] = df['Record'].apply(lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x))

    # Create a figure
    fig = go.Figure()

    # Create color map (excluding noise)
    unique_clusters = sorted([c for c in viz_df['cluster'].unique() if c != -1])
    colorscale = px.colors.qualitative.Bold  # Bold colors for clusters
    colors = {cluster: colorscale[i % len(colorscale)] for i, cluster in enumerate(unique_clusters)}

    # First, add noise points (cluster -1) with less prominence
    noise_points = viz_df[viz_df['cluster'] == -1]
    if len(noise_points) > 0:
        fig.add_trace(go.Scatter(
            x=noise_points['x'],
            y=noise_points['y'],
            mode='markers',
            marker=dict(
                color='lightgrey',
                size=4,
                opacity=0.5
            ),
            name='Unclustered',
            hoverinfo='text',
            hovertext=noise_points.apply(
                lambda row: f"Unclustered<br>Title: {row.get('title', 'N/A')}"
                if 'title' in noise_points.columns else "Unclustered",
                axis=1
            )
        ))

    # Add each cluster as a separate trace
    for cluster in unique_clusters:
        cluster_points = viz_df[viz_df['cluster'] == cluster]

        # Skip if no points in cluster
        if len(cluster_points) == 0:
            continue

        # Get cluster name
        cluster_name = cluster_names.get(cluster, f"Cluster {cluster}") if cluster_names else f"Cluster {cluster}"

        # Add the trace
        fig.add_trace(go.Scatter(
            x=cluster_points['x'],
            y=cluster_points['y'],
            mode='markers',
            marker=dict(
                color=colors[cluster],
                size=8,
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            name=cluster_name,
            hoverinfo='text',
            hovertext=cluster_points.apply(
                lambda row: f"<b>{cluster_name}</b><br>Title: {row.get('title', 'N/A')}"
                            + (f"<br>{row.get('abstract', '')}" if 'abstract' in cluster_points.columns else ""),
                axis=1
            )
        ))

        # Add annotation for cluster centroid
        center_x = cluster_points['x'].mean()
        center_y = cluster_points['y'].mean()

        fig.add_trace(go.Scatter(
            x=[center_x],
            y=[center_y],
            mode='markers+text',
            marker=dict(
                symbol='circle',
                size=30,
                opacity=0.5,
                color=colors[cluster],
                line=dict(width=1, color='white')
            ),
            text=cluster_name,
            textposition="middle center",
            textfont=dict(
                family="Arial",
                size=10,
                color="black"
            ),
            showlegend=False,
            hoverinfo='none'
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False,
            showticklabels=False,
            title=""
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False,
            showticklabels=False,
            title=""
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1,
            itemsizing="constant",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="lightgrey",
            borderwidth=1
        ),
        plot_bgcolor='rgba(240, 240, 250, 0.2)',
        height=700,
        width=900,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='closest'
    )

    # Add cluster statistics in the corner
    stats_text = "<b>Cluster Statistics</b><br>"
    stats_text += f"Total Articles: {len(viz_df)}<br>"
    stats_text += f"Clustered: {len(viz_df[viz_df['cluster'] != -1])} ({len(viz_df[viz_df['cluster'] != -1]) / len(viz_df) * 100:.1f}%)<br>"
    stats_text += f"Unclustered: {len(viz_df[viz_df['cluster'] == -1])} ({len(viz_df[viz_df['cluster'] == -1]) / len(viz_df) * 100:.1f}%)<br>"
    stats_text += f"Number of Clusters: {len(unique_clusters)}"

    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="lightgrey",
        borderwidth=1,
        align="left",
        xanchor="left",
        yanchor="top"
    )

    return fig


def assign_noise_to_nearest_cluster(df, coords_2d):
    """Assigns noise points to their nearest cluster"""
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    result_df = df.copy()

    # Get points that are not noise
    non_noise_mask = result_df['cluster'] != -1
    non_noise_coords = coords_2d[non_noise_mask]
    non_noise_clusters = result_df.loc[non_noise_mask, 'cluster'].values

    # Get noise points
    noise_mask = result_df['cluster'] == -1
    noise_coords = coords_2d[noise_mask]

    # If we have both noise and non-noise points
    if len(non_noise_coords) > 0 and len(noise_coords) > 0:
        # Find nearest non-noise point for each noise point
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(non_noise_coords)
        distances, indices = nn.kneighbors(noise_coords)

        # Assign noise points to nearest cluster
        result_df.loc[noise_mask, 'cluster'] = non_noise_clusters[indices.flatten()]

        # Store original classification (was it noise or not)
        result_df['was_noise'] = noise_mask

    return result_df

