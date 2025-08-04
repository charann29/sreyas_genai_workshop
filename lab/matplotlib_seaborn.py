import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataVisualizer:
    """
    A comprehensive data visualization toolkit for Generative AI datasets
    """
    
    def __init__(self):
        self.fig_size = (12, 8)
        self.colors = sns.color_palette("husl", 10)
    
    def generate_sample_data(self):
        """Generate sample datasets for demonstration"""
        # Generate synthetic tabular data
        X_class, y_class = make_classification(
            n_samples=1000, n_features=10, n_informative=5,
            n_redundant=2, n_clusters_per_class=1, random_state=42
        )
        
        # Generate blob data for clustering visualization
        X_blob, y_blob = make_blobs(
            n_samples=500, centers=4, cluster_std=1.5, random_state=42
        )
        
        # Create a DataFrame for tabular data
        feature_names = [f'feature_{i}' for i in range(X_class.shape[1])]
        df_tabular = pd.DataFrame(X_class, columns=feature_names)
        df_tabular['target'] = y_class
        
        # Generate time series data
        np.random.seed(42)
        time_series = np.cumsum(np.random.randn(1000)) + np.sin(np.linspace(0, 20, 1000))
        
        # Generate text-like data (word frequencies)
        np.random.seed(42)
        vocab_size = 1000
        doc_lengths = np.random.poisson(100, 200)
        word_freqs = [np.random.zipf(1.5, vocab_size)[:length] for length in doc_lengths]
        
        return {
            'tabular': df_tabular,
            'blob_features': X_blob,
            'blob_labels': y_blob,
            'time_series': time_series,
            'word_frequencies': word_freqs
        }
    
    def plot_feature_distributions(self, df, save_path=None):
        """Plot histograms of all numerical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = min(4, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                axes[i].hist(df[col], bins=30, alpha=0.7, color=self.colors[i % len(self.colors)])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, df, save_path=None):
        """Create correlation heatmap for numerical features"""
        numerical_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numerical_df.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_scatter_matrix(self, df, target_col=None, save_path=None):
        """Create scatter plot matrix for feature relationships"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if target_col and target_col in numerical_cols:
            numerical_cols = [col for col in numerical_cols if col != target_col]
        
        # Select top 5 features to avoid overcrowding
        selected_cols = numerical_cols[:5]
        
        if target_col and target_col in df.columns:
            colors = df[target_col]
            scatter_kws = {'c': colors, 'alpha': 0.6, 'cmap': 'viridis'}
        else:
            scatter_kws = {'alpha': 0.6}
        
        pd.plotting.scatter_matrix(df[selected_cols], figsize=(15, 15), 
                                 diagonal='hist', **scatter_kws)
        plt.suptitle('Feature Scatter Matrix', y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distributions(self, df, target_col, save_path=None):
        """Visualize class distributions and feature distributions by class"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        df[target_col].value_counts().plot(kind='bar', ax=axes[0,0], color=self.colors[:len(df[target_col].unique())])
        axes[0,0].set_title('Class Distribution')
        axes[0,0].set_xlabel('Class')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # Box plots for first few numerical features by class
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_col]
        
        if len(numerical_cols) >= 1:
            sns.boxplot(data=df, x=target_col, y=numerical_cols[0], ax=axes[0,1])
            axes[0,1].set_title(f'{numerical_cols[0]} by Class')
        
        if len(numerical_cols) >= 2:
            sns.boxplot(data=df, x=target_col, y=numerical_cols[1], ax=axes[1,0])
            axes[1,0].set_title(f'{numerical_cols[1]} by Class')
        
        # Violin plot for feature distribution by class
        if len(numerical_cols) >= 3:
            sns.violinplot(data=df, x=target_col, y=numerical_cols[2], ax=axes[1,1])
            axes[1,1].set_title(f'{numerical_cols[2]} Distribution by Class')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dimensionality_reduction(self, X, y=None, save_path=None):
        """Visualize data using PCA and t-SNE"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        if y is not None:
            scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=axes[0])
        else:
            axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
        
        axes[0].set_title(f'PCA Visualization\nExplained Variance: {pca.explained_variance_ratio_.sum():.2f}')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        axes[0].grid(True, alpha=0.3)
        
        # t-SNE
        if X.shape[0] > 50:  # t-SNE can be slow on large datasets
            sample_idx = np.random.choice(X.shape[0], 500, replace=False)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx] if y is not None else None
        else:
            X_sample = X
            y_sample = y
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X_sample.shape[0]-1))
        X_tsne = tsne.fit_transform(X_sample)
        
        if y_sample is not None:
            scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=axes[1])
        else:
            axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
        
        axes[1].set_title('t-SNE Visualization')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_time_series_analysis(self, time_series, save_path=None):
        """Analyze and visualize time series data"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original time series
        axes[0,0].plot(time_series, alpha=0.8)
        axes[0,0].set_title('Time Series Data')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Value')
        axes[0,0].grid(True, alpha=0.3)
        
        # Distribution of values
        axes[0,1].hist(time_series, bins=50, alpha=0.7, color=self.colors[1])
        axes[0,1].set_title('Value Distribution')
        axes[0,1].set_xlabel('Value')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True, alpha=0.3)
        
        # Moving average
        window_size = min(50, len(time_series) // 10)
        moving_avg = pd.Series(time_series).rolling(window=window_size).mean()
        axes[1,0].plot(time_series, alpha=0.5, label='Original')
        axes[1,0].plot(moving_avg, alpha=0.8, label=f'Moving Average (window={window_size})')
        axes[1,0].set_title('Time Series with Moving Average')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Value')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Autocorrelation
        from statsmodels.tsa.stattools import acf
        autocorr = acf(time_series, nlags=min(100, len(time_series)//4))
        axes[1,1].plot(autocorr)
        axes[1,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1,1].set_title('Autocorrelation Function')
        axes[1,1].set_xlabel('Lag')
        axes[1,1].set_ylabel('Autocorrelation')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_text_analysis(self, word_frequencies, save_path=None):
        """Analyze and visualize text-like data"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Document length distribution
        doc_lengths = [len(doc) for doc in word_frequencies]
        axes[0,0].hist(doc_lengths, bins=30, alpha=0.7, color=self.colors[2])
        axes[0,0].set_title('Document Length Distribution')
        axes[0,0].set_xlabel('Document Length')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        
        # Word frequency distribution (Zipf's law)
        all_words = [word for doc in word_frequencies for word in doc]
        word_counts = pd.Series(all_words).value_counts()
        ranks = np.arange(1, len(word_counts) + 1)
        
        axes[0,1].loglog(ranks[:100], word_counts.values[:100], 'o-', alpha=0.7)
        axes[0,1].set_title('Word Frequency vs Rank (Zipf\'s Law)')
        axes[0,1].set_xlabel('Rank')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True, alpha=0.3)
        
        # Vocabulary size vs document length
        vocab_sizes = [len(set(doc)) for doc in word_frequencies]
        axes[1,0].scatter(doc_lengths, vocab_sizes, alpha=0.6, color=self.colors[3])
        axes[1,0].set_title('Vocabulary Size vs Document Length')
        axes[1,0].set_xlabel('Document Length')
        axes[1,0].set_ylabel('Unique Words')
        axes[1,0].grid(True, alpha=0.3)
        
        # Top words
        top_words = word_counts.head(20)
        axes[1,1].barh(range(len(top_words)), top_words.values, color=self.colors[4])
        axes[1,1].set_yticks(range(len(top_words)))
        axes[1,1].set_yticklabels([f'word_{w}' for w in top_words.index])
        axes[1,1].set_title('Top 20 Most Frequent Words')
        axes[1,1].set_xlabel('Frequency')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to demonstrate all visualization techniques"""
    visualizer = DataVisualizer()
    
    print("Generating sample datasets...")
    data = visualizer.generate_sample_data()
    
    print("\n1. Feature Distribution Analysis")
    print("="*50)
    visualizer.plot_feature_distributions(data['tabular'])
    
    print("\n2. Correlation Analysis")
    print("="*50)
    visualizer.plot_correlation_heatmap(data['tabular'])
    
    print("\n3. Feature Relationships")
    print("="*50)
    visualizer.plot_scatter_matrix(data['tabular'], 'target')
    
    print("\n4. Class Distribution Analysis")
    print("="*50)
    visualizer.plot_class_distributions(data['tabular'], 'target')
    
    print("\n5. Dimensionality Reduction Visualization")
    print("="*50)
    # Remove target column for dimensionality reduction
    X = data['tabular'].drop('target', axis=1).values
    y = data['tabular']['target'].values
    visualizer.plot_dimensionality_reduction(X, y)
    
    print("\n6. Clustering Visualization")
    print("="*50)
    visualizer.plot_dimensionality_reduction(data['blob_features'], data['blob_labels'])
    
    print("\n7. Time Series Analysis")
    print("="*50)
    visualizer.plot_time_series_analysis(data['time_series'])
    
    print("\n8. Text Data Analysis")
    print("="*50)
    visualizer.plot_text_analysis(data['word_frequencies'])
    
    print("\nVisualization complete! All plots have been generated.")
    print("\nKey insights from the visualizations:")
    print("- Feature distributions help identify skewness and outliers")
    print("- Correlation heatmaps reveal feature relationships")
    print("- Dimensionality reduction shows data structure and clusters")
    print("- Time series analysis reveals trends and patterns")
    print("- Text analysis shows document characteristics and word distributions")

if __name__ == "__main__":
    main()
