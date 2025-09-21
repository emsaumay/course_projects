# Fraud Assignmnet 3
# Saumay Lunawat CS24MTECH14005
# Shivendra Deshpande CS24MTECH12017
# Chinmay Rajesh Ingle CS23MTECH12002

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy.spatial.distance import cdist

# --------------------------------------------------
# DATA PROCESSING MODULE
# --------------------------------------------------

def load_data(file_path):
    """
    Load and preprocess the data.

    Args:
        file_path (str): Path to the CSV data file

    Returns:
        tuple: (original_dataframe, normalized_data, scaler)
    """
    print(f"Loading data from {file_path}...")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Select only numeric columns for analysis
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    data = df[numeric_cols].values

    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    print(f"✓ Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"✓ Selected {len(numeric_cols)} numeric features for analysis")

    return df, normalized_data, scaler

# --------------------------------------------------
# VAE MODEL MODULE
# --------------------------------------------------

class VAE(keras.Model):
    """
    Variational Autoencoder (VAE) implementation.
    """

    def __init__(self, input_dim, latent_dim, hidden_units=[128, 64]):
        """
        Initialize the VAE model.

        Args:
            input_dim (int): Dimensionality of input data
            latent_dim (int): Dimensionality of latent space
            hidden_units (list): List of hidden layer units
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Build encoder and decoder networks
        self.encoder_network = self._build_encoder(input_dim, hidden_units, latent_dim)
        self.decoder_network = self._build_decoder(latent_dim, hidden_units, input_dim)

    def _build_encoder(self, input_dim, hidden_units, latent_dim):
        """Build the encoder part of the VAE."""
        encoder_inputs = keras.Input(shape=(input_dim,))
        x = encoder_inputs

        # Add hidden layers
        for units in hidden_units:
            x = layers.Dense(units, activation='relu')(x)

        # Output mean and log variance of the latent distribution
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)

        # Sampling function for the latent space
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # Apply sampling
        z = layers.Lambda(sampling)([z_mean, z_log_var])

        # Create and return the encoder model
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder

    def _build_decoder(self, latent_dim, hidden_units, output_dim):
        """Build the decoder part of the VAE."""
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = latent_inputs

        # Add hidden layers in reverse order
        for units in reversed(hidden_units):
            x = layers.Dense(units, activation='relu')(x)

        # Output layer - linear activation for continuous data
        decoder_outputs = layers.Dense(output_dim, activation='linear')(x)

        # Create and return the decoder model
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder

    def call(self, inputs):
        """Forward pass through the VAE."""
        z_mean, z_log_var, z = self.encoder_network(inputs)
        reconstructed = self.decoder_network(z)

        # Add KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)

        return reconstructed

    def encode(self, inputs):
        """Encode inputs to the latent space."""
        z_mean, z_log_var, z = self.encoder_network(inputs)
        return z

def train_vae(data, latent_dim, epochs=50, batch_size=32):
    """
    Train a VAE model.

    Args:
        data (ndarray): Normalized input data
        latent_dim (int): Dimensionality of latent space
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training

    Returns:
        tuple: (trained_vae, latent_representations)
    """
    input_dim = data.shape[1]

    # Initialize and compile VAE
    vae = VAE(input_dim, latent_dim)
    vae.compile(optimizer=keras.optimizers.Adam())

    # Train VAE
    print(f"Training VAE with latent dimension: {latent_dim}")
    vae.fit(data, data, epochs=epochs, batch_size=batch_size, verbose=0)

    # Get latent representation
    latent_data = vae.encode(data).numpy()

    return vae, latent_data

# --------------------------------------------------
# CLUSTERING MODULE
# --------------------------------------------------

def find_optimal_clusters(data, max_clusters=10):
    """
    Apply the elbow method to determine the optimal number of clusters.

    Args:
    - data (ndarray): Data to cluster
    - max_clusters (int): Maximum number of clusters to try

    Returns:
    - int: Optimal number of clusters
    """
    print("Determining optimal number of clusters using the elbow method...")

    # Initialize lists to store distortion values and corresponding cluster numbers
    cluster_numbers = range(1, max_clusters + 1)
    distortion_values = []

    # Iterate over the range of cluster numbers
    for k in cluster_numbers:
        # Create a KMeans model with the current cluster number
        kmeans_model = KMeans(n_clusters=k, random_state=42)

        # Fit the model to the data
        kmeans_model.fit(data)

        # Calculate the distortion (sum of squared distances) for the current cluster number
        distortion = sum(np.min(cdist(data, kmeans_model.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0]

        # Append the distortion value to the list
        distortion_values.append(distortion)

    # Print the distortion values for each cluster number
    print("\nDistortion values for each cluster number:")
    for k, dist in zip(cluster_numbers, distortion_values):
        print(f"  k={k}: {dist:.4f}")

    # Calculate the rate of decrease in distortion
    distortion_decreases = np.array([distortion_values[i-1] - distortion_values[i] for i in range(1, len(distortion_values))])
    rates_of_decrease = distortion_decreases / np.array(distortion_values[:-1])

    # Find the elbow point - where the rate of decrease significantly drops
    elbow_point_index = np.argmax(rates_of_decrease < 0.1)
    if elbow_point_index == 0:  # If no clear elbow point is found
        elbow_point = 3  # Default to a reasonable value
    else:
        elbow_point = cluster_numbers[elbow_point_index]

    return elbow_point

def evaluate_clusters(latent_data, n_clusters_range):
    """
    Evaluate different numbers of clusters using silhouette score.

    Args:
        latent_data (ndarray): Latent space representations
        n_clusters_range (range): Range of cluster numbers to evaluate

    Returns:
        tuple: (best_kmeans, best_score, all_results)
    """
    print("\nEvaluating different cluster configurations:")

    best_score = -1
    best_kmeans = None
    results = []

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(latent_data)

        # Calculate silhouette score if more than one cluster
        if n_clusters > 1:
            score = silhouette_score(latent_data, cluster_labels)
            results.append({
                'n_clusters': n_clusters,
                'silhouette_score': score
            })

            print(f"  Clusters: {n_clusters}, Silhouette Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_kmeans = kmeans

    return best_kmeans, best_score, results

# --------------------------------------------------
# OUTLIER DETECTION MODULE
# --------------------------------------------------

def find_outliers(latent_data, kmeans, df, percentile_threshold=95):
    """
    Find outliers using points far from their cluster centers.

    Args:
        latent_data (ndarray): Latent space representations
        kmeans (KMeans): Trained KMeans model
        df (DataFrame): Original dataframe
        percentile_threshold (int): Percentile threshold for outlier detection

    Returns:
        tuple: (outlier_dataframe, outlier_indices, point_distances)
    """
    # Get cluster assignments and centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Calculate distances of each point to its cluster center
    distances = np.zeros(len(latent_data))
    for i in range(len(latent_data)):
        distances[i] = np.linalg.norm(latent_data[i] - centers[labels[i]])

    # Set distance threshold as percentile
    distance_threshold = np.percentile(distances, percentile_threshold)

    # Find boundary points (those with distances > threshold)
    boundary_points = np.where(distances > distance_threshold)[0]

    # Create outlier dataframe
    outlier_df = df.iloc[boundary_points].copy()
    outlier_df['distance_to_center'] = distances[boundary_points]
    outlier_df['cluster'] = labels[boundary_points]

    # Save outliers to CSV file
    outlier_csv_path = 'outliers.csv'
    outlier_df.to_csv(outlier_csv_path, index=True)
    print(f"✓ Outliers saved to '{outlier_csv_path}'")

    return outlier_df, boundary_points, distances

def display_outlier_info(outlier_df, boundary_points, df):
    """
    Display information about detected outliers.

    Args:
        outlier_df (DataFrame): Dataframe containing outliers
        boundary_points (ndarray): Indices of outlier points
        df (DataFrame): Original dataframe
    """
    print("\n" + "="*50)
    print(" OUTLIER DETECTION RESULTS ")
    print("="*50)
    print(f"Total outliers detected: {len(boundary_points)} out of {len(df)} data points")
    print(f"Outlier percentage: {len(boundary_points)/len(df)*100:.2f}%")

    if len(boundary_points) > 0:
        print("\nTop outliers by distance to cluster center:")
        top_outliers = outlier_df.sort_values('distance_to_center', ascending=False).head(5)

        for i, (idx, row) in enumerate(top_outliers.iterrows()):
            print(f"\nOutlier #{i+1} (original index: {idx}):")
            print(f"  Cluster: {int(row['cluster'])}")
            print(f"  Distance to center: {row['distance_to_center']:.4f}")

            # Print original features (excluding distance and cluster)
            print("  Feature values:")
            for col in df.columns:
                print(f"    - {col}: {row[col]}")

# --------------------------------------------------
# VISUALIZATION MODULE
# --------------------------------------------------

def visualize_clusters_and_outliers(latent_data, kmeans, boundary_points, latent_dim):
    """
    Visualize clusters and outliers in the latent space.
    Only 2D visualization is enabled as requested.

    Args:
        latent_data (ndarray): Latent space representations
        kmeans (KMeans): Trained KMeans model
        boundary_points (ndarray): Indices of outlier points
        latent_dim (int): Dimensionality of latent space
    """
    # Only proceed with visualization if latent dimension is 2
    if latent_dim != 2:
        print("\nVisualization requires 2D latent space. Using the best configuration with latent_dim=2 for visualization.")
        return

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 2D visualization
    plt.figure(figsize=(12, 10))

    # Plot all points colored by cluster
    for i in range(len(np.unique(labels))):
        mask = labels == i
        plt.scatter(latent_data[mask, 0], latent_data[mask, 1],
                   label=f'Cluster {i}', alpha=0.6)

    # Mark outliers
    if len(boundary_points) > 0:
        plt.scatter(latent_data[boundary_points, 0], latent_data[boundary_points, 1],
                    s=100, facecolors='none', edgecolors='r', linewidth=2,
                    label='Outliers')

    # Plot cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='X', c='k',
               label='Cluster Centers')

    plt.title('Clusters and Outliers in 2D Latent Space', fontsize=16)
    plt.xlabel('Latent Dimension 1', fontsize=12)
    plt.ylabel('Latent Dimension 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('clusters_2d.png', dpi=300)
    print("\n✓ Visualization saved as 'clusters_2d.png'")
    plt.show()

def visualize_experiment_results(results):
    """
    Placeholder function to maintain compatibility.
    Experiment visualization disabled as requested.

    Args:
        results (list): List of dictionaries with experiment results
    """
    # Visualization disabled as requested
    pass

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_vae_clustering_pipeline(
    file_path,
    latent_dims=[2],  # Only using latent_dim=2 as requested
    percentile_threshold=95,
    epochs=50,
    batch_size=32
):
    """
    Main function to run the complete VAE-based clustering and outlier detection pipeline.

    Args:
        file_path (str): Path to the CSV data file
        latent_dims (list): List of latent dimensions to evaluate
        percentile_threshold (int): Percentile threshold for outlier detection
        epochs (int): Number of VAE training epochs
        batch_size (int): Batch size for VAE training

    Returns:
        tuple: (original_df, best_latent_data, best_kmeans, outlier_df, best_vae)
    """
    print("\n" + "="*50)
    print(" VAE CLUSTERING AND ANOMALY DETECTION ")
    print("="*50)

    # 1. Load and preprocess data
    df, normalized_data, scaler = load_data(file_path)

    # 2. Train VAEs with different latent dimensions
    best_score = -1
    best_config = None
    best_latent_data = None
    best_kmeans = None
    best_vae = None
    all_results = []

    print("\n" + "-"*50)
    print(" EXPERIMENT CONFIGURATIONS ")
    print("-"*50)

    for latent_dim in latent_dims:
        # Train VAE and get latent representation
        vae, latent_data = train_vae(normalized_data, latent_dim, epochs, batch_size)

        # Determine optimal number of clusters
        optimal_clusters = find_optimal_clusters(latent_data)

        # Evaluate a range around the optimal clusters
        n_clusters_range = range(max(2, optimal_clusters-2), optimal_clusters+3)
        kmeans, score, results = evaluate_clusters(latent_data, n_clusters_range)

        # Add latent dimension to results
        for res in results:
            res['latent_dim'] = latent_dim

        all_results.extend(results)

        # Check if this is the best configuration
        if score > best_score:
            best_score = score
            best_config = (latent_dim, kmeans.n_clusters)
            best_latent_data = latent_data
            best_kmeans = kmeans
            best_vae = vae

    print("\n" + "-"*50)
    print(" BEST CONFIGURATION ")
    print("-"*50)
    print(f"Latent Dimension: {best_config[0]}")
    print(f"Number of Clusters: {best_config[1]}")
    print(f"Silhouette Score: {best_score:.4f}")

    # 3. Find outliers with the best configuration
    outlier_df, boundary_points, distances = find_outliers(
        best_latent_data, best_kmeans, df, percentile_threshold)

    # 4. Display outlier information
    display_outlier_info(outlier_df, boundary_points, df)

    # 5. Visualize results
    visualize_clusters_and_outliers(best_latent_data, best_kmeans, boundary_points, best_config[0])
    visualize_experiment_results(all_results)

    print("\n" + "="*50)
    print(" PIPELINE COMPLETED SUCCESSFULLY ")
    print("="*50)

    return df, best_latent_data, best_kmeans, outlier_df, best_vae

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":
    file_path = "data.csv"
    df, latent_data, kmeans, outlier_df, vae = run_vae_clustering_pipeline(file_path)
