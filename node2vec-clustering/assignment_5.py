# Fraud Assignmnet 5
# Saumay Lunawat CS24MTECH14005
# Shivendra Deshpande CS24MTECH12017
# Chinmay Rajesh Ingle CS23MTECH12002

import pandas as pd
import networkx as nx
import numpy as np
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import warnings

def load_data(file_path):
    """Load data from Excel"""
    print(f"Loading data from '{file_path}'...")
    start_time = time.time()
    df = pd.read_excel(file_path)
    print(f"Loaded {len(df)} transactions.")
    df = validate_and_clean_data(df, start_time)
    return df

def validate_and_clean_data(df, start_time):
    """Validate and clean the data."""
    
    # Define required columns
    required_cols = ['Sender', 'Receiver', 'Amount']
    
    # Convert Sender and Receiver to string type
    df['Sender'] = df['Sender'].astype(str)
    df['Receiver'] = df['Receiver'].astype(str)
    
    # Convert Amount to numeric type and handle errors
    df['Amount'] = pd.to_numeric(df['Amount'])
    
    # Convert Amount to float type
    df['Amount'] = df['Amount'].astype(float)
    
    # Print time taken for data loading
    print(f"Data loading finished in {(time.time() - start_time):.2f} seconds.")
    
    return df



def create_and_aggregate_graph(transaction_df: pd.DataFrame):
    """
    Builds a weighted directed network graph from transaction data.

    Aggregates transaction amounts between the same sender and receiver
    into a single weighted edge. Removes self-loops and isolated nodes.
    """
    print("\nBuilding and aggregating the network graph...")

    start_time = time.time()
    network_graph = nx.DiGraph()

    # --- Aggregation Step (Refactored using pandas groupby) ---

    # Prepare DataFrame: handle missing 'Amount' and filter self-loops
    processed_df = transaction_df.copy() # Work on a copy
    if 'Amount' not in processed_df.columns:
        print("Warning: 'Amount' column not found. Using 1.0 for all transaction weights.")
        processed_df['Amount'] = 1.0
    else:
        # Ensure amount is numeric and handle potential NaNs if necessary
        processed_df['Amount'] = pd.to_numeric(processed_df['Amount'], errors='coerce').fillna(0)


    # Filter out self-loops before aggregation
    filtered_transactions = processed_df[processed_df['Sender'] != processed_df['Receiver']]

    # Aggregate edge weights by grouping on (Sender, Receiver) and summing Amount
    # This replaces the explicit loop and defaultdict
    aggregated_weights = filtered_transactions.groupby(['Sender', 'Receiver'])['Amount'].sum()

    # --- Graph Construction ---

    # Add all unique potential nodes first (from original data)
    # This ensures nodes without outgoing/incoming edges in the *filtered*
    # data are still considered initially before isolation removal.
    all_potential_nodes = set(transaction_df['Sender'].unique()).union(set(transaction_df['Receiver'].unique()))
    network_graph.add_nodes_from(all_potential_nodes)


    # Add weighted edges from the aggregated results
    # aggregated_weights is a pandas Series with MultiIndex (Sender, Receiver)
    for (source_node, target_node), total_weight in aggregated_weights.items():
        # Only add edges with positive weight, matching original logic
        if total_weight > 0:
            network_graph.add_edge(source_node, target_node, weight=total_weight)

    # --- Post-Construction Filtering ---

    # Remove nodes that ended up with no connections (in or out) in the final graph
    nodes_to_remove = list(nx.isolates(network_graph))
    if nodes_to_remove:
        network_graph.remove_nodes_from(nodes_to_remove)
        print(f"Removed {len(nodes_to_remove)} isolated nodes.")

    # --- Final Preparations ---

    # Get the list of nodes *after* removal of isolates
    current_nodes = list(network_graph.nodes())
    node_count = network_graph.number_of_nodes()

    # Create mappings between node labels (original identifiers) and integer indices
    node_label_to_index = {label: i for i, label in enumerate(current_nodes)}
    index_to_node_label = {i: label for label, i in node_label_to_index.items()}

    # --- Validation and Output ---

    print(f"Weighted directed graph built/aggregated. Nodes: {node_count}, Edges: {network_graph.number_of_edges()}")

    if node_count < 2:
        print("Error: Graph has less than 2 nodes after processing. Cannot proceed.")
        sys.exit(1) # Use sys.exit()

    if network_graph.number_of_edges() == 0:
        print("Error: Graph has no edges after processing. Cannot perform downstream analysis like Node2Vec or clustering.")
        sys.exit(1) # Use sys.exit()

    print(f"Graph processing finished in {time.time() - start_time:.2f} seconds.")

    return network_graph, current_nodes, node_label_to_index, index_to_node_label, node_count

# --- Alias Method for Sampling ---
def alias_setup(probs):
    """Setup for non-uniform sampling."""
    K = len(probs)
    if K == 0: return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)
    smaller, larger = [], []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0: smaller.append(kk)
        else: larger.append(kk)
    while len(smaller) > 0 and len(larger) > 0:
        small, large = smaller.pop(), larger.pop()
        J[small] = large
        q[large] += q[small] - 1.0
        if q[large] < 1.0: smaller.append(large)
        else: larger.append(large)
    q[q > 1.0] = 1.0 # Handle potential numerical issues
    return J, q

def alias_draw(J, q):
    """Draw sample using alias method."""
    K = len(J)
    if K == 0: return -1
    kk = int(np.floor(np.random.rand() * K))
    return kk if np.random.rand() < q[kk] else J[kk]

# --- Biased Random Walk ---
def get_alias_edge(graph, src, dst, p, q):
    """Get alias setup for transitions from dst, given src."""
    unnormalized_probs = []
    neighbors_dst = list(graph.neighbors(dst))
    if not neighbors_dst: return alias_setup([]), []

    considered_neighbors = []
    for dst_neighbor in neighbors_dst:
        weight = graph[dst][dst_neighbor].get('weight', 1.0)
        if weight <= 0: continue
        considered_neighbors.append(dst_neighbor)
        if dst_neighbor == src: unnormalized_probs.append(weight / p)
        elif graph.has_edge(src, dst_neighbor): unnormalized_probs.append(weight)
        else: unnormalized_probs.append(weight / q)

    norm_const = sum(unnormalized_probs)
    if norm_const > 0:
        normalized_probs = [u_prob / norm_const for u_prob in unnormalized_probs]
    else:
        normalized_probs = []
    return alias_setup(normalized_probs), considered_neighbors

def simulate_walks(graph, num_walks, walk_length, p, q, nodes_list, node_to_idx):
    """Simulate biased random walks on the graph."""
    print(f"  Simulating {num_walks} walks of length {walk_length} for each node...")
    start_time = time.time()
    walks = []
    nodes_list_shuffled = list(nodes_list)

    # Precompute transition probabilities
    alias_nodes = {}
    alias_edges = {}
    print("  Precomputing transition probabilities...")
    for node in graph.nodes():
        out_neighbors = list(graph.neighbors(node))
        considered_out_neighbors_node = [neighbor for neighbor in out_neighbors if graph[node][neighbor].get('weight', 1.0) > 0]
        unnormalized_probs_node = [graph[node][neighbor].get('weight', 1.0) for neighbor in considered_out_neighbors_node]
        norm_const_node = sum(unnormalized_probs_node)
        if norm_const_node > 0:
            normalized_probs_node = [u_prob / norm_const_node for u_prob in unnormalized_probs_node]
            alias_nodes[node] = (alias_setup(normalized_probs_node), considered_out_neighbors_node)
        else:
            alias_nodes[node] = (alias_setup([]), [])

        for neighbor in considered_out_neighbors_node:
             alias_table, neighbor_neighbors = get_alias_edge(graph, node, neighbor, p, q)
             alias_edges[(node, neighbor)] = (alias_table, neighbor_neighbors)

    # Simulate walks
    walk_count = 0
    total_walks_expected = len(nodes_list) * num_walks
    for walk_iter in range(num_walks):
        random.shuffle(nodes_list_shuffled)
        for start_node in nodes_list_shuffled:
            walk = [start_node]
            while len(walk) < walk_length:
                cur = walk[-1]
                if len(walk) == 1:
                    (J, q_alias), current_neighbors = alias_nodes.get(cur, (np.array([], dtype=np.int32), []))
                else:
                    prev = walk[-2]
                    (J, q_alias), current_neighbors = alias_edges.get((prev, cur), (np.array([], dtype=np.int32), []))

                if not current_neighbors: break
                next_node_idx_in_neighbors = alias_draw(J, q_alias)
                if next_node_idx_in_neighbors != -1:
                    walk.append(current_neighbors[next_node_idx_in_neighbors])
                else:
                    break # Should not happen if current_neighbors is not empty

            if len(walk) > 1:
                 walks.append([node_to_idx[node] for node in walk])
            walk_count += 1
            if walk_count % 1000 == 0 or walk_count == total_walks_expected:
                 elapsed = time.time() - start_time
                 est_total_time = (elapsed / walk_count * total_walks_expected) if walk_count > 0 else 0
                 print(f"    Generated {walk_count}/{total_walks_expected} walks... (Est. total time: {est_total_time:.1f}s)", end='\r')

    print(f"\n    Walk simulation finished. Generated {len(walks)} valid walks.")
    return walks

# --- Skip-gram Model ---
class SkipGramModel(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super(SkipGramModel, self).__init__()
        self.in_embed = nn.Embedding(num_nodes, embed_dim)
        self.out_embed = nn.Embedding(num_nodes, embed_dim)
        nn.init.kaiming_uniform_(self.in_embed.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.out_embed.weight, nonlinearity='relu')

    def forward(self, target_nodes, context_nodes, negative_nodes):
        target_emb = self.in_embed(target_nodes)
        context_emb = self.out_embed(context_nodes.squeeze(1))
        negative_emb = self.out_embed(negative_nodes)

        pos_dot = torch.sum(target_emb * context_emb, dim=1)
        log_pos = torch.log(torch.sigmoid(pos_dot) + 1e-6)

        neg_dot = torch.bmm(negative_emb, target_emb.unsqueeze(2)).squeeze(2)
        log_neg = torch.log(torch.sigmoid(-neg_dot) + 1e-6).sum(dim=1)

        loss = -(log_pos + log_neg)
        return loss.mean()

def generate_batch(walks, window_size, batch_size, num_nodes, num_negative):
    """Generator for Skip-gram training batches."""
    all_pairs = []
    for walk in walks:
        for i, target_node_idx in enumerate(walk):
            current_window = random.randint(1, window_size)
            start = max(0, i - current_window)
            end = min(len(walk), i + current_window + 1)
            context_indices = [walk[j] for j in range(start, end) if i != j]
            for context_idx in context_indices:
                all_pairs.append((target_node_idx, context_idx))

    if not all_pairs: return # No pairs to yield

    random.shuffle(all_pairs)
    idx = 0
    while idx < len(all_pairs):
        batch_targets, batch_contexts, batch_negatives = [], [], []
        count = 0
        while count < batch_size and idx < len(all_pairs):
            target, context = all_pairs[idx]
            idx += 1
            neg_samples = []
            attempts = 0
            while len(neg_samples) < num_negative and attempts < num_negative * 5:
                neg_candidate = random.randint(0, num_nodes - 1)
                if neg_candidate != target and neg_candidate != context and neg_candidate not in neg_samples:
                    neg_samples.append(neg_candidate)
                attempts += 1
            if len(neg_samples) == num_negative:
                batch_targets.append(target)
                batch_contexts.append(context)
                batch_negatives.append(neg_samples)
                count += 1
        if batch_targets:
            yield (torch.LongTensor(batch_targets),
                   torch.LongTensor(batch_contexts).unsqueeze(1),
                   torch.LongTensor(batch_negatives))

def train_node2vec(walks, num_nodes, embed_dim, window_size, num_negative, batch_size, learning_rate, epochs):
    """Train the Skip-gram model on generated walks."""
    print(f"  Starting Node2Vec training ({epochs} epochs)...")
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    model = SkipGramModel(num_nodes, embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        epoch_start_time = time.time()
        batch_generator = generate_batch(walks, window_size, batch_size, num_nodes, num_negative)

        # Check if batches are available
        try:
            peek = next(batch_generator)
            batch_generator = generate_batch(walks, window_size, batch_size, num_nodes, num_negative) # Recreate generator
            has_batches = True
        except StopIteration:
            has_batches = False
            print(f"    Epoch {epoch+1}/{epochs}: No training batches generated. Skipping epoch.")
            continue

        for i, (targets, contexts, negatives) in enumerate(batch_generator):
            targets, contexts, negatives = targets.to(device), contexts.to(device), negatives.to(device)
            optimizer.zero_grad()
            loss = model(targets, contexts, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            if (i + 1) % 200 == 0:
                 print(f"    Epoch {epoch+1}/{epochs}, Batch {i+1}, Current Avg Loss: {total_loss / (i + 1):.4f}", end='\r')

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        epoch_time = time.time() - epoch_start_time
        print(f"\n    Epoch {epoch+1}/{epochs} complete. Batches processed: {batch_count}, Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    print("  Node2Vec training finished.")
    print(f"Training finished in {time.time() - start_time:.2f} seconds.")

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        node_embeddings = model.in_embed.weight.cpu().numpy()
    print(f"Generated node embeddings of shape: {node_embeddings.shape}")
    return node_embeddings

def find_optimal_k_elbow(embeddings, k_range, num_nodes):
    """Find optimal K using the Elbow method."""
    print(f"\nFinding optimal K using Elbow method (testing K={list(k_range)})...")
    start_time = time.time()

    # Adjust K range based on number of nodes
    max_k = min(num_nodes - 1, k_range[-1])
    if max_k < k_range.start:
         print(f"Error: Not enough nodes ({num_nodes}) for the specified K range ({k_range.start}-{k_range[-1]}).")
         return None # Indicate failure

    range_n_clusters_adjusted = range(k_range.start, max_k + 1)
    inertia_values = []
    print(f"  Testing K in range: {list(range_n_clusters_adjusted)}")

    for n_clusters in range_n_clusters_adjusted:
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertia_values.append(kmeans.inertia_)
            print(f"    K={n_clusters}, Inertia (WCSS): {kmeans.inertia_:.2f}")
        except Exception as e:
             print(f"    An error occurred for K={n_clusters}: {e}")
             inertia_values.append(np.nan) # Mark as invalid

    # Find the elbow point (simple heuristic)
    optimal_k = k_range.start
    points = np.array([(k, inertia_values[i]) for i, k in enumerate(range_n_clusters_adjusted) if not np.isnan(inertia_values[i])])

    if len(points) >= 3:
        first_point, last_point = points[0], points[-1]
        m = (last_point[1] - first_point[1]) / (last_point[0] - first_point[0])
        A, B, C = m, -1, first_point[1] - m * first_point[0]
        distances = []
        for i in range(1, len(points) - 1):
            k, inertia = points[i]
            distance = abs(A * k + B * inertia + C) / np.sqrt(A**2 + B**2)
            distances.append(distance)
        if distances:
            max_distance_index = np.argmax(distances)
            optimal_k = int(points[max_distance_index + 1][0])

    print(f"\nEstimated optimal K via Elbow heuristic: {optimal_k}")
    print(f"Elbow analysis finished in {time.time() - start_time:.2f} seconds.")
    return optimal_k, range_n_clusters_adjusted, inertia_values


def perform_final_clustering(embeddings, n_clusters, num_nodes, idx_to_node):
    """Perform K-Means clustering with the chosen number of clusters."""
    # Ensure N_CLUSTERS is valid
    final_n_clusters = max(2, min(n_clusters, num_nodes))
    if final_n_clusters != n_clusters:
         print(f"Warning: Adjusted final N_CLUSTERS from {n_clusters} to {final_n_clusters} due to node count.")
    n_clusters = final_n_clusters

    print(f"\nPerforming final K-Means clustering with N_CLUSTERS = {n_clusters}...")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    # Map clusters back to original node IDs
    node_clusters = {idx_to_node[i]: cluster for i, cluster in enumerate(clusters)}

    print(f"Final clustering finished in {time.time() - start_time:.2f} seconds.")
    return clusters, node_clusters, n_clusters

def visualize_clusters(embeddings, clusters, num_nodes, idx_to_node, n_clusters, graph_stats, sample_labels):
    """Visualize clusters using PCA."""
    print("\nVisualizing clusters using PCA...")
    start_time = time.time()
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    explained_variance = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA reduction complete. Explained variance by 2 components: {explained_variance:.2f}%")

    print("  Plotting...")
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7, s=50)

    # Add sample labels
    if sample_labels > 0 and num_nodes > 0:
        indices_to_label = random.sample(range(num_nodes), min(sample_labels, num_nodes))
        for i in indices_to_label:
            node_id = idx_to_node[i]
            plt.text(embeddings_2d[i, 0] + 0.05, embeddings_2d[i, 1] + 0.05, str(node_id), fontsize=8, alpha=0.8)

    plt.title(f'Node2Vec Embeddings Clustered (K={n_clusters}) using PCA\n(Nodes: {graph_stats["nodes"]}, Edges: {graph_stats["edges"]})')
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

    # Create legend
    cluster_labels = [f'Cluster {i}' for i in range(n_clusters)]
    handles, _ = scatter.legend_elements()
    if len(handles) == n_clusters and n_clusters > 0:
        plt.legend(handles=handles, labels=cluster_labels, title="Clusters")
    elif n_clusters > 0:
         print("Warning: Could not create precise legend. Using colorbar.")
         plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = "cluster_visualization.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()  # Show the plot in a window
    print(f"Visualization finished in {time.time() - start_time:.2f} seconds.")

def plot_elbow_curve(k_range, inertia_values, optimal_k):
    """Plot the Elbow method curve."""
    plt.figure(figsize=(10, 7))
    plt.plot(list(k_range), inertia_values, marker='o')
    plt.title('Elbow Method: Inertia vs. Number of Clusters (K)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.xticks(list(k_range))
    plt.grid(True)
    if optimal_k in k_range:
        elbow_index = list(k_range).index(optimal_k)
        plt.scatter(k_range[elbow_index], inertia_values[elbow_index], color='red', s=200, marker='*', label=f'Estimated Elbow (K={optimal_k})')
        plt.legend()
    plt.show()
    plt.savefig("elbow_method_plot.png", dpi=300)


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Node2Vec Clustering Assignment (Elbow Method) ---")

    # Configuration
    PAYMENTS_FILE = "payments.xlsx"
    EMBEDDING_DIM = 64
    WALK_LENGTH = 40
    NUM_WALKS = 15
    P = 1.0
    Q = 1.0
    WINDOW_SIZE = 5
    NUM_NEGATIVE = 5
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    EPOCHS = 5
    K_RANGE_FOR_ELBOW = range(2, 7)
    PLOT_SAMPLE_LABELS = 0

    # 1. Load Data
    df = load_data(PAYMENTS_FILE)

    # 2. Create and Aggregate Graph
    graph, nodes, node_to_idx, idx_to_node, num_nodes = create_and_aggregate_graph(df)
    graph_stats = {"nodes": num_nodes, "edges": graph.number_of_edges()}

    # 3. Node2Vec Implementation and Training
    walks = simulate_walks(graph, NUM_WALKS, WALK_LENGTH, P, Q, nodes, node_to_idx)
    if not walks:
        print("Node2Vec walk generation failed. Exiting.")
        exit()
    node_embeddings = train_node2vec(walks, num_nodes, EMBEDDING_DIM, WINDOW_SIZE, NUM_NEGATIVE, BATCH_SIZE, LEARNING_RATE, EPOCHS)

    # 4. Find Optimal Clusters using Elbow Method
    optimal_k, k_range_adjusted, inertia_values = find_optimal_k_elbow(node_embeddings, K_RANGE_FOR_ELBOW, num_nodes)
    if optimal_k is None:
         print("Could not determine optimal K. Exiting.")
         exit()

    # Plot Elbow curve
    plot_elbow_curve(k_range_adjusted, inertia_values, optimal_k)

    # 5. Final Clustering with Optimal K
    clusters, node_clusters, N_CLUSTERS = perform_final_clustering(node_embeddings, optimal_k, num_nodes, idx_to_node)

    # 6. Visualization with PCA
    visualize_clusters(node_embeddings, clusters, num_nodes, idx_to_node, N_CLUSTERS, graph_stats, PLOT_SAMPLE_LABELS)