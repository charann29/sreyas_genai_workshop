# Experiment 3: Data Visualization for Generative AI Datasets
## Cell-by-Cell Implementation with Explanations

---

## Cell 1: Import Libraries and Setup
**What**: Import all necessary libraries for data visualization and analysis  
**Why**: We need matplotlib/seaborn for plotting, pandas/numpy for data handling, and sklearn for sample data generation

```python
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
print("Libraries imported successfully!")
```

---

## Cell 2: Generate Sample Datasets
**What**: Create synthetic datasets that mimic real Generative AI data scenarios  
**Why**: We need realistic sample data to demonstrate visualization techniques - includes tabular, clustering, time series, and text-like data

```python
def generate_sample_data():
    """Generate comprehensive sample datasets for demonstration"""
    
    # 1. Tabular data with classification target
    X_class, y_class = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_redundant=2, n_clusters_per_class=1, random_state=42
    )
    
    # 2. Clustering data (for unsupervised learning visualization)
    X_blob, y_blob = make_blobs(
        n_samples=500, centers=4, cluster_std=1.5, random_state=42
    )
    
    # 3. Create DataFrame for tabular analysis
    feature_names = [f'feature_{i}' for i in range(X_class.shape[1])]
    df_tabular = pd.DataFrame(X_class, columns=feature_names)
    df_tabular['target'] = y_class
    
    # 4. Time series data (for sequential data analysis)
    np.random.seed(42)
    time_series = np.cumsum(np.random.randn(1000)) + np.sin(np.linspace(0, 20, 1000))
    
    # 5. Text-like data (word frequencies following Zipf's law)
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

# Generate all sample datasets
data = generate_sample_data()
print("Sample datasets generated:")
print(f"- Tabular data shape: {data['tabular'].shape}")
print(f"- Blob data shape: {data['blob_features'].shape}")
print(f"- Time series length: {len(data['time_series'])}")
print(f"- Number of documents: {len(data['word_frequencies'])}")
```

---

## Cell 3: Feature Distribution Analysis
**What**: Create histograms showing the distribution of each numerical feature  
**Why**: Understanding feature distributions helps identify skewness, outliers, and data quality issues before training models

```python
def plot_feature_distributions(df, title_suffix=""):
    """Plot histograms of all numerical features"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = min(4, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    colors = sns.color_palette("husl", len(numerical_cols))
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            axes[i].hist(df[col], bins=30, alpha=0.7, color=colors[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = df[col].mean()
            std_val = df[col].std()
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            axes[i].legend()
    
    # Hide unused subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Feature Distributions{title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.show()

# Execute visualization
plot_feature_distributions(data['tabular'], " - Tabular Dataset")
```

---

## Cell 4: Correlation Analysis
**What**: Create a heatmap showing correlations between all numerical features  
**Why**: Identifies multicollinearity and feature relationships, crucial for feature selection and understanding data structure

```python
def plot_correlation_heatmap(df):
    """Create correlation heatmap for numerical features"""
    numerical_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numerical_df.corr()
    
    plt.figure(figsize=(12, 10))
    
    # Create mask for upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
               center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
               fmt='.2f')
    
    plt.title('Feature Correlation Heatmap\n(Red = Positive, Blue = Negative)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Print insights
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                      correlation_matrix.columns[j], corr_val))
    
    if high_corr_pairs:
        print("High correlation pairs (|r| > 0.7):")
        for pair in high_corr_pairs:
            print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
    else:
        print("No high correlation pairs found (|r| > 0.7)")

# Execute correlation analysis
plot_correlation_heatmap(data['tabular'])
```

---

## Cell 5: Feature Relationships (Scatter Matrix)
**What**: Create scatter plots showing pairwise relationships between features  
**Why**: Reveals non-linear relationships, clusters, and patterns that correlation might miss

```python
def plot_scatter_matrix(df, target_col=None):
    """Create scatter plot matrix for feature relationships"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if target_col and target_col in numerical_cols:
        numerical_cols = [col for col in numerical_cols if col != target_col]
    
    # Select top 5 features to avoid overcrowding
    selected_cols = numerical_cols[:5]
    
    fig, axes = plt.subplots(len(selected_cols), len(selected_cols), 
                           figsize=(15, 15))
    
    for i, col1 in enumerate(selected_cols):
        for j, col2 in enumerate(selected_cols):
            if i == j:
                # Diagonal: histograms
                axes[i, j].hist(df[col1], bins=20, alpha=0.7)
                axes[i, j].set_title(f'{col1}')
            else:
                # Off-diagonal: scatter plots
                if target_col and target_col in df.columns:
                    scatter = axes[i, j].scatter(df[col2], df[col1], 
                                               c=df[target_col], alpha=0.6, 
                                               cmap='viridis', s=10)
                else:
                    axes[i, j].scatter(df[col2], df[col1], alpha=0.6, s=10)
                
                axes[i, j].set_xlabel(col2)
                axes[i, j].set_ylabel(col1)
            
            axes[i, j].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Scatter Matrix (Color = Target Class)', fontsize=16)
    plt.tight_layout()
    plt.show()

# Execute scatter matrix
plot_scatter_matrix(data['tabular'], 'target')
```

---

## Cell 6: Class Distribution Analysis
**What**: Analyze target variable distribution and feature differences across classes  
**Why**: Class imbalance detection and understanding how features vary by class is crucial for classification tasks

```python
def plot_class_distributions(df, target_col):
    """Visualize class distributions and feature distributions by class"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Class distribution bar chart
    class_counts = df[target_col].value_counts()
    axes[0,0].bar(range(len(class_counts)), class_counts.values, 
                  color=sns.color_palette("husl", len(class_counts)))
    axes[0,0].set_title('Class Distribution')
    axes[0,0].set_xlabel('Class')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_xticks(range(len(class_counts)))
    axes[0,0].set_xticklabels(class_counts.index)
    
    # Add percentage labels
    total = class_counts.sum()
    for i, v in enumerate(class_counts.values):
        axes[0,0].text(i, v + total*0.01, f'{v/total*100:.1f}%', 
                      ha='center', va='bottom')
    
    # 2-4. Box plots for first three numerical features by class
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != target_col]
    
    plot_positions = [(0,1), (1,0), (1,1)]
    
    for idx, col in enumerate(numerical_cols[:3]):
        row, col_idx = plot_positions[idx]
        
        # Box plot
        sns.boxplot(data=df, x=target_col, y=col, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'{col} Distribution by Class')
        axes[row, col_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print class statistics
    print("Class Distribution Summary:")
    for class_val in df[target_col].unique():
        count = (df[target_col] == class_val).sum()
        percentage = count / len(df) * 100
        print(f"  Class {class_val}: {count} samples ({percentage:.1f}%)")

# Execute class analysis
plot_class_distributions(data['tabular'], 'target')
```

---

## Cell 7: Dimensionality Reduction Visualization
**What**: Use PCA and t-SNE to visualize high-dimensional data in 2D  
**Why**: Reveals data structure, clusters, and separability that's impossible to see in high dimensions

```python
def plot_dimensionality_reduction(X, y=None, title_suffix=""):
    """Visualize data using PCA and t-SNE"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. PCA Analysis
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    if y is not None:
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                                cmap='viridis', alpha=0.7, s=30)
        plt.colorbar(scatter, ax=axes[0])
    else:
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=30)
    
    # Add explained variance information
    total_var = pca.explained_variance_ratio_.sum()
    axes[0].set_title(f'PCA Visualization{title_suffix}\n'
                     f'Total Explained Variance: {total_var:.1%}')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    axes[0].grid(True, alpha=0.3)
    
    # 2. t-SNE Analysis (sample data if too large)
    if X.shape[0] > 500:
        sample_idx = np.random.choice(X.shape[0], 500, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx] if y is not None else None
        sample_note = " (500 samples)"
    else:
        X_sample = X
        y_sample = y
        sample_note = ""
    
    # Standardize for t-SNE
    X_scaled = StandardScaler().fit_transform(X_sample)
    tsne = TSNE(n_components=2, random_state=42, 
                perplexity=min(30, X_sample.shape[0]-1))
    X_tsne = tsne.fit_transform(X_scaled)
    
    if y_sample is not None:
        scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, 
                                cmap='viridis', alpha=0.7, s=30)
        plt.colorbar(scatter, ax=axes[1])
    else:
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, s=30)
    
    axes[1].set_title(f't-SNE Visualization{title_suffix}{sample_note}')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"PCA Results:")
    print(f"  - Component 1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
    print(f"  - Component 2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")
    print(f"  - Total explained: {total_var:.1%}")

# Execute dimensionality reduction on tabular data
X_tabular = data['tabular'].drop('target', axis=1).values
y_tabular = data['tabular']['target'].values
plot_dimensionality_reduction(X_tabular, y_tabular, " - Tabular Data")
```

---

## Cell 8: Clustering Data Visualization
**What**: Visualize blob/clustering dataset to show unsupervised learning patterns  
**Why**: Demonstrates how to visualize clustering results and natural data groupings

```python
# Execute dimensionality reduction on clustering data
print("Clustering Dataset Analysis:")
plot_dimensionality_reduction(data['blob_features'], data['blob_labels'], " - Clustering Data")

# Additional clustering analysis
plt.figure(figsize=(12, 5))

# Original 2D clustering data
plt.subplot(1, 2, 1)
scatter = plt.scatter(data['blob_features'][:, 0], data['blob_features'][:, 1], 
                     c=data['blob_labels'], cmap='viridis', alpha=0.7, s=50)
plt.colorbar(scatter)
plt.title('Original Clustering Data\n(2D Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)

# Cluster statistics
plt.subplot(1, 2, 2)
unique_labels, counts = np.unique(data['blob_labels'], return_counts=True)
plt.bar(unique_labels, counts, color=sns.color_palette("viridis", len(unique_labels)))
plt.title('Cluster Size Distribution')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Points')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Cluster Statistics:")
for label, count in zip(unique_labels, counts):
    percentage = count / len(data['blob_labels']) * 100
    print(f"  Cluster {label}: {count} points ({percentage:.1f}%)")
```

---

## Cell 9: Time Series Analysis
**What**: Comprehensive time series visualization including trends, distributions, and autocorrelation  
**Why**: Time series data is common in sequential generation tasks - understanding temporal patterns is crucial

```python
def plot_time_series_analysis(time_series):
    """Analyze and visualize time series data"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Original time series
    axes[0,0].plot(time_series, alpha=0.8, linewidth=1)
    axes[0,0].set_title('Time Series Data')
    axes[0,0].set_xlabel('Time Step')
    axes[0,0].set_ylabel('Value')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(range(len(time_series)), time_series, 1)
    p = np.poly1d(z)
    axes[0,0].plot(range(len(time_series)), p(range(len(time_series))), 
                   "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
    axes[0,0].legend()
    
    # 2. Distribution of values
    axes[0,1].hist(time_series, bins=50, alpha=0.7, color='skyblue', density=True)
    axes[0,1].set_title('Value Distribution')
    axes[0,1].set_xlabel('Value')
    axes[0,1].set_ylabel('Density')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(time_series)
    std_val = np.std(time_series)
    axes[0,1].axvline(mean_val, color='red', linestyle='-', 
                     label=f'Mean: {mean_val:.2f}')
    axes[0,1].axvline(mean_val + std_val, color='red', linestyle='--', 
                     label=f'±1 Std: {std_val:.2f}')
    axes[0,1].axvline(mean_val - std_val, color='red', linestyle='--')
    axes[0,1].legend()
    
    # 3. Moving averages
    window_sizes = [10, 50, 100]
    axes[1,0].plot(time_series, alpha=0.3, label='Original', linewidth=0.5)
    
    colors = ['red', 'green', 'blue']
    for window, color in zip(window_sizes, colors):
        if window < len(time_series):
            moving_avg = pd.Series(time_series).rolling(window=window, center=True).mean()
            axes[1,0].plot(moving_avg, alpha=0.8, label=f'MA({window})', 
                          color=color, linewidth=2)
    
    axes[1,0].set_title('Moving Averages')
    axes[1,0].set_xlabel('Time Step')
    axes[1,0].set_ylabel('Value')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Autocorrelation (simplified version)
    max_lag = min(100, len(time_series) // 4)
    autocorr = []
    
    for lag in range(max_lag):
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
            autocorr.append(corr if not np.isnan(corr) else 0)
    
    axes[1,1].plot(range(max_lag), autocorr, 'b-', alpha=0.8)
    axes[1,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1,1].axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='±0.2 threshold')
    axes[1,1].axhline(y=-0.2, color='r', linestyle='--', alpha=0.5)
    axes[1,1].set_title('Autocorrelation Function')
    axes[1,1].set_xlabel('Lag')
    axes[1,1].set_ylabel('Autocorrelation')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("Time Series Summary:")
    print(f"  Length: {len(time_series)} points")
    print(f"  Mean: {np.mean(time_series):.3f}")
    print(f"  Std: {np.std(time_series):.3f}")
    print(f"  Min: {np.min(time_series):.3f}")
    print(f"  Max: {np.max(time_series):.3f}")
    print(f"  Trend slope: {z[0]:.6f}")

# Execute time series analysis
plot_time_series_analysis(data['time_series'])
```

---

## Cell 10: Text Data Analysis
**What**: Analyze text-like data including document lengths, word frequencies, and vocabulary patterns  
**Why**: Text data visualization is crucial for NLP and text generation tasks - helps understand corpus characteristics

```python
def plot_text_analysis(word_frequencies):
    """Analyze and visualize text-like data"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Document length distribution
    doc_lengths = [len(doc) for doc in word_frequencies]
    axes[0,0].hist(doc_lengths, bins=30, alpha=0.7, color='lightcoral')
    axes[0,0].set_title('Document Length Distribution')
    axes[0,0].set_xlabel('Number of Words')
    axes[0,0].set_ylabel('Number of Documents')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add statistics
    mean_len = np.mean(doc_lengths)
    axes[0,0].axvline(mean_len, color='red', linestyle='--', 
                     label=f'Mean: {mean_len:.1f}')
    axes[0,0].legend()
    
    # 2. Word frequency distribution (Zipf's law)
    all_words = [word for doc in word_frequencies for word in doc]
    word_counts = pd.Series(all_words).value_counts()
    ranks = np.arange(1, min(101, len(word_counts) + 1))
    
    axes[0,1].loglog(ranks, word_counts.values[:len(ranks)], 'o-', alpha=0.7, markersize=4)
    axes[0,1].set_title('Word Frequency vs Rank\n(Zipf\'s Law)')
    axes[0,1].set_xlabel('Rank (log scale)')
    axes[0,1].set_ylabel('Frequency (log scale)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add Zipf's law reference line
    zipf_line = word_counts.values[0] / ranks
    axes[0,1].loglog(ranks, zipf_line, 'r--', alpha=0.5, label='Ideal Zipf')
    axes[0,1].legend()
    
    # 3. Vocabulary size vs document length
    vocab_sizes = [len(set(doc)) for doc in word_frequencies]
    axes[1,0].scatter(doc_lengths, vocab_sizes, alpha=0.6, color='mediumseagreen', s=30)
    axes[1,0].set_title('Vocabulary Size vs Document Length')
    axes[1,0].set_xlabel('Document Length (words)')
    axes[1,0].set_ylabel('Unique Words (vocabulary)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(doc_lengths, vocab_sizes, 1)
    p = np.poly1d(z)
    axes[1,0].plot(sorted(doc_lengths), p(sorted(doc_lengths)), 
                   "r--", alpha=0.8, label=f'Trend (R²: {np.corrcoef(doc_lengths, vocab_sizes)[0,1]**2:.3f})')
    axes[1,0].legend()
    
    # 4. Top frequent words
    top_words = word_counts.head(15)
    y_pos = np.arange(len(top_words))
    
    axes[1,1].barh(y_pos, top_words.values, color='gold', alpha=0.8)
    axes[1,1].set_yticks(y_pos)
    axes[1,1].set_yticklabels([f'word_{w}' for w in top_words.index])
    axes[1,1].set_title('Top 15 Most Frequent Words')
    axes[1,1].set_xlabel('Frequency')
    axes[1,1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # Print corpus statistics
    total_words = len(all_words)
    unique_words = len(word_counts)
    
    print("Text Corpus Summary:")
    print(f"  Total documents: {len(word_frequencies)}")
    print(f"  Total words: {total_words:,}")
    print(f"  Unique words: {unique_words:,}")
    print(f"  Vocabulary richness: {unique_words/total_words:.3f}")
    print(f"  Average document length: {np.mean(doc_lengths):.1f} words")
    print(f"  Average vocabulary per document: {np.mean(vocab_sizes):.1f} words")
    
    # Show top 10 words
    print(f"\nTop 10 most frequent words:")
    for i, (word, freq) in enumerate(word_counts.head(10).items()):
        percentage = freq / total_words * 100
        print(f"  {i+1}. word_{word}: {freq} ({percentage:.2f}%)")

# Execute text analysis
plot_text_analysis(data['word_frequencies'])
```

---

## Cell 11: Summary and Key Insights
**What**: Summarize all visualization insights and their implications for Generative AI  
**Why**: Consolidates learning and explains how these analyses inform model development

```python
print("="*70)
print("EXPERIMENT 3 SUMMARY: DATA VISUALIZATION FOR GENERATIVE AI")
print("="*70)

print("\n📊 VISUALIZATION TECHNIQUES DEMONSTRATED:")
print("1. ✅ Feature Distribution Analysis - Histograms with statistics")
print("2. ✅ Correlation Analysis - Heatmaps for feature relationships") 
print("3. ✅ Scatter Plot Matrix - Pairwise feature relationships")
print("4. ✅ Class Distribution Analysis - Target variable analysis")
print("5. ✅ Dimensionality Reduction - PCA and t-SNE visualizations")
print("6. ✅ Clustering Visualization - Unsupervised pattern detection")
print("7. ✅ Time Series Analysis - Temporal pattern detection")
print("8. ✅ Text Data Analysis - Corpus characteristics and word patterns")

print("\n🔍 KEY INSIGHTS FOR GENERATIVE AI:")
print("\n📈 DATA QUALITY INSIGHTS:")
print("• Feature distributions reveal data skewness and outlier presence")
print("• Correlation analysis identifies redundant features and multicollinearity")
print("• Class distribution shows potential imbalance issues")

print("\n🎯 MODEL DEVELOPMENT INSIGHTS:")
print("• PCA shows which features capture most variance")
print("• t-SNE reveals natural data clusters and separability")
print("• Time series patterns inform sequence modeling approaches")
print("• Text analysis reveals vocabulary size and distribution patterns")

print("\n🚀 PRACTICAL APPLICATIONS:")
print("• Image GANs: Pixel intensity distributions and spatial correlations")
print("• Text Generation: Vocabulary richness and document length patterns")
print("• Sequence Models: Temporal dependencies and autocorrelation")
print("• Data Preprocessing: Normalization needs and feature selection")

print("\n💡 NEXT STEPS:")
print("• Use insights to inform preprocessing decisions")
print("• Select appropriate model architectures based on data structure")
print("• Design data augmentation strategies based on distributions")
print("• Monitor training with similar visualizations")

print("\n" + "="*70)
print("EXPERIMENT 3 COMPLETED SUCCESSFULLY! 🎉")
print("="*70)
```

---

## Cell 12: Save Sample Outputs (Optional)
**What**: Save visualizations and export sample data for further analysis  
**Why**: Allows saving results for reports and sharing with team members

```python
# Optional: Save sample data and create export functions

def save_sample_data():
    """Save generated datasets for future use"""
    # Save tabular data
    data['tabular'].to_csv('sample_tabular_data.csv', index=False)
    
    # Save blob clustering data
    blob_df = pd.DataFrame(data['blob_features'], columns=['feature_0', 'feature_1'])
    blob_df['cluster'] = data['blob_labels']
    blob_df.to_csv('sample_clustering_data.csv', index=False)
    
    # Save time series
    ts_df = pd.DataFrame({'time_step': range(len(data['time_series'])), 
                         'value': data['time_series']})
    ts_df.to_csv('sample_time_series.csv', index=False)
    
    print("Sample datasets saved:")
    print("• sample_tabular_data.csv - Classification dataset")
    print("• sample_clustering_data.csv - Clustering dataset") 
    print("• sample_time_series.csv - Time series data")

def create_visualization_functions():
    """Create reusable visualization functions for your own data"""
    
    template_code = '''
# REUSABLE VISUALIZATION TEMPLATE
# Copy this code to analyze your own datasets

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def quick_data_overview(df, target_col=None):
    """Quick overview of any dataset"""
    print("Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\\nColumn types:")
    print(df.dtypes.value_counts())
    
    if target_col and target_col in df.columns:
        print(f"\\nTarget variable '{target_col}':")
        print(df[target_col].value_counts())
    
    print("\\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found")

def plot_data_summary(df, target_col=None):
    """Create summary visualizations for any dataset"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Missing data heatmap
    sns.heatmap(df.isnull(), cbar=True, ax=axes[0,0])
    axes[0,0].set_title('Missing Data Pattern')
    
    # 2. Correlation heatmap (numerical features only)
    numerical_df = df.select_dtypes(include=[np.number])
    if len(numerical_df.columns) > 1:
        sns.heatmap(numerical_df.corr(), annot=True, cmap='RdBu_r', 
                   center=0, ax=axes[0,1])
        axes[0,1].set_title('Feature Correlations')
    
    # 3. Feature distributions
    if len(numerical_df.columns) > 0:
        first_feature = numerical_df.columns[0]
        axes[1,0].hist(df[first_feature].dropna(), bins=30, alpha=0.7)
        axes[1,0].set_title(f'Distribution: {first_feature}')
    
    # 4. Target distribution (if provided)
    if target_col and target_col in df.columns:
        df[target_col].value_counts().plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title(f'Target Distribution: {target_col}')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Usage example:
# df = pd.read_csv('your_data.csv')
# quick_data_overview(df, 'target_column')
# plot_data_summary(df, 'target_column')
'''
    
    with open('visualization_template.py', 'w') as f:
        f.write(template_code)
    
    print("Visualization template saved as 'visualization_template.py'")
    print("You can use this template to analyze your own datasets!")

# Execute save functions
save_sample_data()
create_visualization_functions()

print("\n" + "="*50)
print("FILES CREATED FOR FUTURE USE:")
print("="*50)
print("📁 sample_tabular_data.csv")
print("📁 sample_clustering_data.csv") 
print("📁 sample_time_series.csv")
print("📁 visualization_template.py")
print("\nYou can now use these files to practice with real data!")
```

---

## Cell 13: Advanced Visualization Techniques (Bonus)
**What**: Additional advanced visualization techniques for specialized Generative AI applications  
**Why**: These techniques are useful for specific generative modeling scenarios and research

```python
def advanced_visualizations():
    """Advanced visualization techniques for specialized use cases"""
    
    print("🚀 ADVANCED VISUALIZATION TECHNIQUES")
    print("="*50)
    
    # 1. Feature Importance Visualization
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    
    # Train a quick model for feature importance
    X = data['tabular'].drop('target', axis=1)
    y = data['tabular']['target']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    feature_importance = rf.feature_importances_
    indices = np.argsort(feature_importance)[::-1]
    
    plt.bar(range(len(feature_importance)), feature_importance[indices])
    plt.xticks(range(len(feature_importance)), 
               [X.columns[i] for i in indices], rotation=45)
    plt.title('Random Forest Feature Importance')
    plt.ylabel('Importance Score')
    
    # 2. Learning Curve Simulation
    plt.subplot(1, 2, 2)
    # Simulate learning curves
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = 1 - np.exp(-train_sizes * 3) + np.random.normal(0, 0.02, len(train_sizes))
    val_scores = 1 - np.exp(-train_sizes * 2.5) + np.random.normal(0, 0.03, len(train_sizes))
    
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score', alpha=0.8)
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score', alpha=0.8)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Simulated Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Data Quality Metrics Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Completeness
    completeness = (1 - data['tabular'].isnull().sum() / len(data['tabular'])) * 100
    axes[0,0].bar(range(len(completeness)), completeness.values)
    axes[0,0].set_title('Data Completeness (%)')
    axes[0,0].set_xticks(range(len(completeness)))
    axes[0,0].set_xticklabels(completeness.index, rotation=45)
    axes[0,0].axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
    axes[0,0].legend()
    
    # Outlier detection (using IQR method)
    numerical_cols = data['tabular'].select_dtypes(include=[np.number]).columns
    outlier_counts = []
    
    for col in numerical_cols:
        Q1 = data['tabular'][col].quantile(0.25)
        Q3 = data['tabular'][col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data['tabular'][col] < (Q1 - 1.5 * IQR)) | 
                   (data['tabular'][col] > (Q3 + 1.5 * IQR))).sum()
        outlier_counts.append(outliers)
    
    axes[0,1].bar(range(len(outlier_counts)), outlier_counts)
    axes[0,1].set_title('Outlier Count (IQR Method)')
    axes[0,1].set_xticks(range(len(outlier_counts)))
    axes[0,1].set_xticklabels(numerical_cols, rotation=45)
    
    # Skewness
    from scipy.stats import skew
    skewness_values = [skew(data['tabular'][col].dropna()) for col in numerical_cols]
    
    colors = ['red' if abs(s) > 1 else 'orange' if abs(s) > 0.5 else 'green' 
              for s in skewness_values]
    
    axes[0,2].bar(range(len(skewness_values)), skewness_values, color=colors, alpha=0.7)
    axes[0,2].set_title('Feature Skewness')
    axes[0,2].set_xticks(range(len(skewness_values)))
    axes[0,2].set_xticklabels(numerical_cols, rotation=45)
    axes[0,2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0,2].axhline(y=1, color='red', linestyle='--', alpha=0.5)
    axes[0,2].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    
    # Variance across features
    variances = [data['tabular'][col].var() for col in numerical_cols]
    axes[1,0].bar(range(len(variances)), variances)
    axes[1,0].set_title('Feature Variances')
    axes[1,0].set_xticks(range(len(variances)))
    axes[1,0].set_xticklabels(numerical_cols, rotation=45)
    axes[1,0].set_yscale('log')
    
    # Feature ranges (min-max spread)
    ranges = [data['tabular'][col].max() - data['tabular'][col].min() for col in numerical_cols]
    axes[1,1].bar(range(len(ranges)), ranges)
    axes[1,1].set_title('Feature Ranges (Max - Min)')
    axes[1,1].set_xticks(range(len(ranges)))
    axes[1,1].set_xticklabels(numerical_cols, rotation=45)
    
    # Unique value counts
    unique_counts = [data['tabular'][col].nunique() for col in data['tabular'].columns]
    axes[1,2].bar(range(len(unique_counts)), unique_counts)
    axes[1,2].set_title('Unique Value Counts')
    axes[1,2].set_xticks(range(len(unique_counts)))
    axes[1,2].set_xticklabels(data['tabular'].columns, rotation=45)
    
    plt.suptitle('Data Quality Dashboard', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("✅ Advanced visualizations completed!")
    print("\n📊 Data Quality Summary:")
    print(f"• Average completeness: {completeness.mean():.1f}%")
    print(f"• Total outliers detected: {sum(outlier_counts)}")
    print(f"• Highly skewed features (|skew| > 1): {sum(1 for s in skewness_values if abs(s) > 1)}")
    print(f"• Features with high variance: {sum(1 for v in variances if v > np.mean(variances) * 2)}")

# Execute advanced visualizations
advanced_visualizations()
```

---

## Cell 14: Interactive Visualization (Optional)
**What**: Create interactive plots using plotly for dynamic exploration  
**Why**: Interactive plots allow deeper data exploration and are great for presentations

```python
# Note: This cell requires plotly. Install with: pip install plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    def create_interactive_visualizations():
        """Create interactive visualizations using Plotly"""
        
        print("🎯 CREATING INTERACTIVE VISUALIZATIONS")
        print("="*50)
        
        # 1. Interactive Scatter Plot with PCA
        from sklearn.decomposition import PCA
        
        X = data['tabular'].drop('target', axis=1)
        y = data['tabular']['target']
        
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        
        # Create DataFrame for plotly
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1], 
            'PC3': X_pca[:, 2],
            'Target': y.astype(str),
            'Index': range(len(y))
        })
        
        # 3D PCA plot
        fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', 
                           color='Target', hover_data=['Index'],
                           title='Interactive 3D PCA Visualization')
        fig.show()
        
        # 2. Interactive correlation heatmap
        numerical_df = data['tabular'].select_dtypes(include=[np.number])
        corr_matrix = numerical_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Interactive Correlation Heatmap',
            xaxis_title='Features',
            yaxis_title='Features'
        )
        fig.show()
        
        # 3. Interactive time series
        ts_df = pd.DataFrame({
            'Time': range(len(data['time_series'])),
            'Value': data['time_series']
        })
        
        fig = px.line(ts_df, x='Time', y='Value', 
                     title='Interactive Time Series Plot')
        fig.add_hline(y=ts_df['Value'].mean(), line_dash="dash", 
                     annotation_text="Mean")
        fig.show()
        
        print("✅ Interactive visualizations created!")
        print("💡 Tip: Use mouse to zoom, pan, and hover for details")
    
    create_interactive_visualizations()
    
except ImportError:
    print("📝 INTERACTIVE VISUALIZATIONS (PLOTLY REQUIRED)")
    print("="*50)
    print("To create interactive visualizations, install plotly:")
    print("pip install plotly")
    print("\nThen uncomment and run the interactive visualization code above.")
    print("Interactive plots are great for:")
    print("• 3D data exploration")
    print("• Zooming into specific data regions") 
    print("• Hovering for detailed information")
    print("• Sharing dynamic visualizations")
```

---

## Cell 15: Real-World Application Examples
**What**: Show how to apply these techniques to specific Generative AI use cases  
**Why**: Connects the visualization techniques to practical applications in different domains

```python
def generative_ai_applications():
    """Examples of how to apply visualizations to specific Generative AI tasks"""
    
    print("🎨 GENERATIVE AI APPLICATION EXAMPLES")
    print("="*70)
    
    examples = {
        "IMAGE GENERATION (GANs)": {
            "data_types": ["Pixel intensities", "Color distributions", "Spatial features"],
            "key_visualizations": [
                "• Pixel intensity histograms per channel (RGB)",
                "• Spatial correlation heatmaps",
                "• Generated vs. real image feature distributions",
                "• Latent space visualization (t-SNE of generator inputs)",
                "• Training loss curves (Generator vs. Discriminator)"
            ],
            "insights": [
                "→ Identify mode collapse in generated images",
                "→ Monitor training stability",
                "→ Understand color bias in generated samples"
            ]
        },
        
        "TEXT GENERATION (LLMs)": {
            "data_types": ["Token sequences", "Word embeddings", "Attention patterns"],
            "key_visualizations": [
                "• Vocabulary frequency distributions (Zipf's law)",
                "• Sentence length histograms",
                "• Word embedding scatter plots (PCA/t-SNE)",
                "• N-gram frequency analysis",
                "• Perplexity curves during training"
            ],
            "insights": [
                "→ Detect vocabulary bias and coverage",
                "→ Monitor text quality metrics",
                "→ Identify repetitive patterns in generation"
            ]
        },
        
        "MUSIC GENERATION": {
            "data_types": ["Note sequences", "Rhythm patterns", "Harmonic features"],
            "key_visualizations": [
                "• Note distribution histograms",
                "• Temporal pattern autocorrelation",
                "• Chord progression frequency analysis",
                "• Rhythm pattern clustering",
                "• Pitch range distributions"
            ],
            "insights": [
                "→ Balance between different musical keys",
                "→ Temporal consistency in generated sequences",
                "→ Harmonic complexity analysis"
            ]
        },
        
        "TIME SERIES GENERATION": {
            "data_types": ["Sequential values", "Temporal patterns", "Seasonality"],
            "key_visualizations": [
                "• Autocorrelation function analysis",
                "• Seasonal decomposition plots",
                "• Distribution comparison (real vs. generated)",
                "• Spectral density analysis",
                "• Rolling statistics visualization"
            ],
            "insights": [
                "→ Preserve temporal dependencies",
                "→ Maintain statistical properties",
                "→ Detect overfitting to training patterns"
            ]
        }
    }
    
    for application, details in examples.items():
        print(f"\n🎯 {application}")
        print("-" * len(application))
        
        print(f"📊 Data Types: {', '.join(details['data_types'])}")
        
        print("\n🔍 Essential Visualizations:")
        for viz in details['key_visualizations']:
            print(f"  {viz}")
        
        print("\n💡 Key Insights:")
        for insight in details['insights']:
            print(f"  {insight}")
        
        print()
    
    # Practical code example for monitoring GAN training
    print("\n" + "="*70)
    print("📝 PRACTICAL EXAMPLE: MONITORING GAN TRAINING")
    print("="*70)
    
    monitoring_code = '''
def monitor_gan_training(real_samples, generated_samples, epoch):
    """Monitor GAN training progress with visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Feature distribution comparison
    for i, feature_idx in enumerate([0, 1, 2]):  # First 3 features
        if i < 3:
            axes[0, i].hist(real_samples[:, feature_idx], alpha=0.5, 
                           label='Real', bins=30, density=True)
            axes[0, i].hist(generated_samples[:, feature_idx], alpha=0.5, 
                           label='Generated', bins=30, density=True)
            axes[0, i].set_title(f'Feature {feature_idx} Distribution')
            axes[0, i].legend()
    
    # 2. 2D scatter comparison
    axes[1, 0].scatter(real_samples[:500, 0], real_samples[:500, 1], 
                      alpha=0.6, label='Real', s=20)
    axes[1, 0].scatter(generated_samples[:500, 0], generated_samples[:500, 1], 
                      alpha=0.6, label='Generated', s=20)
    axes[1, 0].set_title('2D Feature Space')
    axes[1, 0].legend()
    
    # 3. Statistical comparison
    stats_real = [np.mean(real_samples, axis=0), np.std(real_samples, axis=0)]
    stats_gen = [np.mean(generated_samples, axis=0), np.std(generated_samples, axis=0)]
    
    x_pos = np.arange(len(stats_real[0][:5]))  # First 5 features
    axes[1, 1].bar(x_pos - 0.2, stats_real[0][:5], 0.4, label='Real Mean')
    axes[1, 1].bar(x_pos + 0.2, stats_gen[0][:5], 0.4, label='Generated Mean')
    axes[1, 1].set_title('Mean Comparison')
    axes[1, 1].legend()
    
    # 4. Standard deviation comparison
    axes[1, 2].bar(x_pos - 0.2, stats_real[1][:5], 0.4, label='Real Std')
    axes[1, 2].bar(x_pos + 0.2, stats_gen[1][:5], 0.4, label='Generated Std')
    axes[1, 2].set_title('Standard Deviation Comparison')
    axes[1, 2].legend()
    
    plt.suptitle(f'GAN Training Monitor - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Calculate and return quality metrics
    from scipy.stats import wasserstein_distance
    
    quality_metrics = {}
    for i in range(min(5, real_samples.shape[1])):
        wd = wasserstein_distance(real_samples[:, i], generated_samples[:, i])
        quality_metrics[f'wasserstein_dist_feature_{i}'] = wd
    
    return quality_metrics

# Usage during training:
# metrics = monitor_gan_training(real_data, generated_data, current_epoch)
# print(f"Quality metrics: {metrics}")
'''
    
    print(monitoring_code)
    
    print("\n💡 BEST PRACTICES FOR GENERATIVE AI VISUALIZATION:")
    print("="*60)
    best_practices = [
        "🔄 Monitor distributions continuously during training",
        "📊 Compare generated vs. real data statistics regularly", 
        "🎯 Use domain-specific metrics (e.g., FID for images)",
        "⚡ Create automated visualization pipelines",
        "🔍 Visualize failure cases and edge cases",
        "📈 Track quality metrics over training epochs",
        "🎨 Use appropriate color schemes for different data types",
        "💾 Save visualizations for training documentation"
    ]
    
    for practice in best_practices:
        print(f"  {practice}")

# Execute the application examples
generative_ai_applications()
```

---

## Cell 16: Final Exercise and Next Steps
**What**: Provide a hands-on exercise and roadmap for continued learning  
**Why**: Reinforces learning and provides clear next steps for applying these techniques

```python
print("🎓 FINAL EXERCISE: ANALYZE YOUR OWN DATA")
print("="*60)

exercise_instructions = '''
📋 EXERCISE INSTRUCTIONS:

1. 📁 LOAD YOUR DATA:
   - Use pd.read_csv('your_file.csv') for CSV files
   - Or create synthetic data with specific characteristics
   
2. 🔍 APPLY ALL VISUALIZATION TECHNIQUES:
   - Feature distributions
   - Correlation analysis  
   - Dimensionality reduction
   - Class/target analysis (if applicable)
   - Time series analysis (if sequential data)
   
3. 📝 DOCUMENT YOUR FINDINGS:
   - Data quality issues
   - Feature relationships
   - Preprocessing needs
   - Model architecture recommendations

4. 🎯 DOMAIN-SPECIFIC ANALYSIS:
   - Apply relevant techniques for your data type
   - Consider generative modeling implications
'''

print(exercise_instructions)

# Provide a template for the exercise
exercise_template = '''
# YOUR DATA ANALYSIS TEMPLATE
# Fill in with your own dataset

def analyze_my_data():
    """Complete analysis of your dataset"""
    
    # 1. Load your data
    # df = pd.read_csv('your_dataset.csv')
    # print(f"Dataset shape: {df.shape}")
    
    # 2. Quick overview
    # print(df.info())
    # print(df.describe())
    
    # 3. Apply visualizations
    # plot_feature_distributions(df)
    # plot_correlation_heatmap(df)
    # plot_class_distributions(df, 'target_column')  # if applicable
    
    # 4. Advanced analysis
    # X = df.drop('target', axis=1).values  # if applicable
    # y = df['target'].values  # if applicable
    # plot_dimensionality_reduction(X, y)
    
    # 5. Document findings
    findings = {
        'data_quality': 'Your assessment here',
        'key_patterns': 'What patterns did you find?',
        'preprocessing_needs': 'What preprocessing is needed?',
        'model_recommendations': 'What models would work well?'
    }
    
    return findings

# Uncomment and modify to use:
# my_findings = analyze_my_data()
# print("My Analysis Results:", my_findings)
'''

print("\n📝 EXERCISE TEMPLATE:")
print(exercise_template)

print("\n🚀 NEXT STEPS IN YOUR GENERATIVE AI JOURNEY:")
print("="*50)

next_steps = {
    "📚 LEARN MORE ABOUT": [
        "• Advanced statistical tests for data analysis",
        "• Domain-specific visualization techniques", 
        "• Interactive dashboard creation (Streamlit, Dash)",
        "• Automated data profiling tools (pandas-profiling)"
    ],
    
    "🔧 PRACTICAL APPLICATIONS": [
        "• Integrate visualizations into ML pipelines",
        "• Create monitoring dashboards for model training",
        "• Build data quality assessment tools",
        "• Develop automated reporting systems"
    ],
    
    "🎯 SPECIFIC TO GENERATIVE AI": [
        "• Study GAN training visualization techniques",
        "• Learn about latent space visualization", 
        "• Explore evaluation metrics for generated content",
        "• Understand mode collapse detection methods"
    ],
    
    "📖 RECOMMENDED RESOURCES": [
        "• 'The Grammar of Graphics' by Leland Wilkinson",
        "• 'Data Visualization: A Practical Introduction' by Kieran Healy",
        "• Seaborn and Matplotlib documentation",
        "• Papers on GAN evaluation and visualization"
    ]
}

for category, items in next_steps.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  {item}")

print(f"\n🎉 CONGRATULATIONS!")
print("="*30)
print("You've completed Experiment 3: Data Visualization for Generative AI!")
print("\nKey achievements:")
print("✅ Mastered 8 essential visualization techniques")
print("✅ Learned to analyze different data types") 
print("✅ Understood quality assessment methods")
print("✅ Connected visualizations to practical AI applications")
print("\n🚀 You're now ready to visualize and understand any dataset!")
print("   Apply these skills to your next generative AI project!")
```
