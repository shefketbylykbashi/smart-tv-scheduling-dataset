"""
Dimensionality reduction with color-coded instance groups.

Groups instances by source:
- Red:   Instances_IPTV   (prefix: iptv-epg)
- Blue:  Instances_PW     (prefix: epg_)
- Green: Instances_Youtube (prefix: YT)

Usage:
    python dataset_diversity.py <features_csv> <output_folder>

Example:
    python dataset_diversity.py ./features.csv ./plots_grouped
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# UMAP import with fallback
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Instance group definitions
INSTANCE_GROUPS = {
    'Instances_IPTV': {
        'prefixes': ['iptv-epg', 'iptv_epg', 'iptv'],
        'color': 'red',
        'marker': 'o',
        'label': 'IPTV'
    },
    'Instances_PW': {
        'prefixes': ['epg_'],
        'color': 'blue',
        'marker': 's',
        'label': 'PW (EPG)'
    },
    'Instances_Youtube': {
        'prefixes': ['YT_', 'YT-', 'youtube'],
        'color': 'green',
        'marker': '^',
        'label': 'YouTube'
    }
}


def classify_instance(instance_name: str) -> str:
    """
    Classify an instance into a group based on its name prefix.

    Returns group name or 'Unknown' if no match.
    """
    instance_lower = instance_name.lower()

    for group_name, group_info in INSTANCE_GROUPS.items():
        for prefix in group_info['prefixes']:
            if instance_lower.startswith(prefix.lower()):
                return group_name

    return 'Unknown'


def load_features(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """
    Load features from CSV file and classify instances.

    Returns:
        - Full dataframe
        - Feature matrix (numpy array)
        - Instance names
        - Instance groups
    """
    df = pd.read_csv(csv_path)

    # Extract instance names
    instance_names = df['instance_name'].tolist()

    # Classify each instance
    instance_groups = [classify_instance(name) for name in instance_names]

    # Extract feature columns (all except instance_name)
    feature_cols = [col for col in df.columns if col != 'instance_name']
    feature_matrix = df[feature_cols].values

    # Print summary
    print(f"Loaded {len(instance_names)} instances with {len(feature_cols)} features")
    print(f"\nInstance group distribution:")
    group_counts = pd.Series(instance_groups).value_counts()
    for group, count in group_counts.items():
        print(f"  {group}: {count}")

    return df, feature_matrix, instance_names, instance_groups


def perform_pca(features: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    """Perform PCA dimensionality reduction."""
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)

    print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

    return reduced, pca


def perform_tsne(features: np.ndarray, n_components: int = 2, perplexity: int = None) -> np.ndarray:
    """Perform t-SNE dimensionality reduction."""
    n_samples = features.shape[0]

    # Adjust perplexity based on sample size
    if perplexity is None:
        perplexity = min(30, max(5, n_samples // 4))

    print(f"\nt-SNE with perplexity={perplexity}")

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        learning_rate='auto',
        init='pca'
    )
    reduced = tsne.fit_transform(features)

    return reduced


def perform_umap(features: np.ndarray, n_components: int = 2, n_neighbors: int = None) -> np.ndarray:
    """Perform UMAP dimensionality reduction."""
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")

    n_samples = features.shape[0]

    # Adjust n_neighbors based on sample size
    if n_neighbors is None:
        n_neighbors = min(15, max(2, n_samples // 5))

    print(f"\nUMAP with n_neighbors={n_neighbors}")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42
    )
    reduced = reducer.fit_transform(features)

    return reduced


def get_group_properties(group_name: str) -> tuple[str, str, str]:
    """Get color, marker, and label for a group."""
    if group_name in INSTANCE_GROUPS:
        info = INSTANCE_GROUPS[group_name]
        return info['color'], info['marker'], info['label']
    else:
        return 'gray', 'x', 'Unknown'


def create_grouped_plot(
        reduced_data: np.ndarray,
        instance_names: list[str],
        instance_groups: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        output_path: str,
        figsize: tuple = (12, 10)
) -> None:
    """Create and save a 2D scatter plot with color-coded groups."""
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique groups
    unique_groups = sorted(set(instance_groups))

    # Plot each group separately
    for group in unique_groups:
        # Get indices for this group
        indices = [i for i, g in enumerate(instance_groups) if g == group]

        if not indices:
            continue

        # Get group properties
        color, marker, label = get_group_properties(group)

        # Extract data for this group
        group_data = reduced_data[indices]

        ax.scatter(
            group_data[:, 0],
            group_data[:, 1],
            c=color,
            marker=marker,
            alpha=0.7,
            s=100,
            edgecolors='white',
            linewidth=0.5,
            label=f'{label} (n={len(indices)})'
        )

    # Add labels for each point (if not too many)
    if len(instance_names) <= 30:
        for i, name in enumerate(instance_names):
            ax.annotate(
                name[:20] + '...' if len(name) > 20 else name,
                (reduced_data[i, 0], reduced_data[i, 1]),
                fontsize=6,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )

    # Styling
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {output_path}")


def create_combined_grouped_plot(
        pca_data: np.ndarray,
        tsne_data: np.ndarray,
        umap_data: np.ndarray,
        instance_names: list[str],
        instance_groups: list[str],
        output_path: str
) -> None:
    """Create a combined figure with all three DR methods, color-coded by group."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    data_list = [pca_data, tsne_data, umap_data]
    titles = ['PCA', 't-SNE', 'UMAP']
    xlabels = ['PC1', 't-SNE 1', 'UMAP 1']
    ylabels = ['PC2', 't-SNE 2', 'UMAP 2']

    # Get unique groups
    unique_groups = sorted(set(instance_groups))

    for ax, data, title, xlabel, ylabel in zip(axes, data_list, titles, xlabels, ylabels):
        # Plot each group
        for group in unique_groups:
            indices = [i for i, g in enumerate(instance_groups) if g == group]

            if not indices:
                continue

            color, marker, label = get_group_properties(group)
            group_data = data[indices]

            ax.scatter(
                group_data[:, 0],
                group_data[:, 1],
                c=color,
                marker=marker,
                alpha=0.7,
                s=80,
                edgecolors='white',
                linewidth=0.5,
                label=f'{label} (n={len(indices)})'
            )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    plt.suptitle('Dimensionality Reduction by Instance Group',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved combined plot: {output_path}")


def save_reduced_coordinates(
        pca_data: np.ndarray,
        tsne_data: np.ndarray,
        umap_data: np.ndarray,
        instance_names: list[str],
        instance_groups: list[str],
        output_path: str
) -> None:
    """Save reduced coordinates with group labels to CSV."""
    df = pd.DataFrame({
        'instance_name': instance_names,
        'group': instance_groups,
        'pca_1': pca_data[:, 0],
        'pca_2': pca_data[:, 1],
        'tsne_1': tsne_data[:, 0],
        'tsne_2': tsne_data[:, 1],
        'umap_1': umap_data[:, 0],
        'umap_2': umap_data[:, 1]
    })
    df.to_csv(output_path, index=False)
    print(f"Saved coordinates: {output_path}")


def print_group_statistics(
        pca_data: np.ndarray,
        tsne_data: np.ndarray,
        umap_data: np.ndarray,
        instance_groups: list[str]
) -> None:
    """Print statistics for each group in each reduced space."""
    unique_groups = sorted(set(instance_groups))

    print("\n" + "=" * 60)
    print("Group Statistics in Reduced Spaces")
    print("=" * 60)

    for method_name, data in [('PCA', pca_data), ('t-SNE', tsne_data), ('UMAP', umap_data)]:
        print(f"\n{method_name}:")
        for group in unique_groups:
            indices = [i for i, g in enumerate(instance_groups) if g == group]
            if indices:
                group_data = data[indices]
                centroid = group_data.mean(axis=0)
                spread = group_data.std(axis=0)
                print(f"  {group}:")
                print(f"    Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f})")
                print(f"    Spread (std): ({spread[0]:.3f}, {spread[1]:.3f})")


def main():
    parser = argparse.ArgumentParser(
        description='Perform dimensionality reduction with color-coded instance groups.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Instance Groups:
  - Red:   IPTV instances    (prefix: iptv-epg, iptv_epg, iptv)
  - Blue:  PW instances      (prefix: epg_)
  - Green: YouTube instances (prefix: YT_, YT-, youtube)

Outputs:
  - pca_grouped.png:      PCA scatter plot by group
  - tsne_grouped.png:     t-SNE scatter plot by group
  - umap_grouped.png:     UMAP scatter plot by group
  - combined_grouped.png: All three methods side-by-side
  - reduced_coordinates_grouped.csv: 2D coordinates with group labels
        """
    )

    parser.add_argument(
        'features_csv',
        type=str,
        help='Path to features CSV file'
    )

    parser.add_argument(
        'output_folder',
        type=str,
        help='Path to output folder for plots'
    )

    args = parser.parse_args()

    # Create output directory
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Load features and classify instances
        print("=" * 60)
        print("Loading features and classifying instances...")
        print("=" * 60)
        df, features, instance_names, instance_groups = load_features(args.features_csv)

        # Check minimum sample size
        n_samples = features.shape[0]
        if n_samples < 3:
            print(f"Warning: Only {n_samples} samples. Results may not be meaningful.")

        # Perform PCA
        print("\n" + "=" * 60)
        print("Performing PCA...")
        print("=" * 60)
        pca_data, pca_model = perform_pca(features)

        create_grouped_plot(
            pca_data,
            instance_names,
            instance_groups,
            title='PCA - TV Scheduling Instances by Group',
            xlabel=f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%} variance)',
            ylabel=f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%} variance)',
            output_path=str(output_folder / 'pca_grouped.png')
        )

        # Perform t-SNE
        print("\n" + "=" * 60)
        print("Performing t-SNE...")
        print("=" * 60)
        tsne_data = perform_tsne(features)

        create_grouped_plot(
            tsne_data,
            instance_names,
            instance_groups,
            title='t-SNE - TV Scheduling Instances by Group',
            xlabel='t-SNE Dimension 1',
            ylabel='t-SNE Dimension 2',
            output_path=str(output_folder / 'tsne_grouped.png')
        )

        # Perform UMAP
        print("\n" + "=" * 60)
        print("Performing UMAP...")
        print("=" * 60)
        if not UMAP_AVAILABLE:
            print("UMAP not available. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                                   'umap-learn', '--break-system-packages', '-q'])
            import umap

        umap_data = perform_umap(features)

        create_grouped_plot(
            umap_data,
            instance_names,
            instance_groups,
            title='UMAP - TV Scheduling Instances by Group',
            xlabel='UMAP Dimension 1',
            ylabel='UMAP Dimension 2',
            output_path=str(output_folder / 'umap_grouped.png')
        )

        # Create combined plot
        print("\n" + "=" * 60)
        print("Creating combined plot...")
        print("=" * 60)
        create_combined_grouped_plot(
            pca_data, tsne_data, umap_data,
            instance_names, instance_groups,
            str(output_folder / 'combined_grouped.png')
        )

        # Print group statistics
        print_group_statistics(pca_data, tsne_data, umap_data, instance_groups)

        # Save coordinates
        save_reduced_coordinates(
            pca_data, tsne_data, umap_data,
            instance_names, instance_groups,
            str(output_folder / 'reduced_coordinates_grouped.csv')
        )

        print("\n" + "=" * 60)
        print("Dimensionality reduction complete!")
        print("=" * 60)
        print(f"\nOutput files in: {output_folder}")
        print("  - pca_grouped.png")
        print("  - tsne_grouped.png")
        print("  - umap_grouped.png")
        print("  - combined_grouped.png")
        print("  - reduced_coordinates_grouped.csv")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
