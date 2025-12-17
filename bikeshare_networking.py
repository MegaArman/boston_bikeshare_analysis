import pandas as pd
import networkx as nx

pd.set_option("display.precision", 3)

# --- 0. Load your parquet (or skip if df already exists) ---
df = pd.read_parquet("network_data_2024_08_to_2025_08.parquet")

# Keep only the columns we need and drop missing station ids
edges_df = (
    df[["start_station_id", "end_station_id", "start_station_name"]]
    .dropna()
    .drop_duplicates()   # important: many trips between same stations
)

# Cast to string just in case ids are mixed types
edges_df["start_station_id"] = edges_df["start_station_id"].astype(str)
edges_df["end_station_id"]   = edges_df["end_station_id"].astype(str)

# --- 1. Build directed graph ---
G = nx.DiGraph()
G.add_edges_from(edges_df[["start_station_id", "end_station_id"]].itertuples(index=False, name=None))

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print(f"Graph has {num_nodes} nodes and {num_edges} unique directed edges.\n")

# Helper to summarize components
def summarize_components(components, label, total_nodes, top_n=10):
    comp_sizes = [len(c) for c in components]
    comp_sizes_sorted = sorted(comp_sizes, reverse=True)

    print(f"=== {label} ===")
    print(f"Number of {label.lower()}: {len(comp_sizes_sorted)}")

    # Show top N components by size
    print(f"Top {min(top_n, len(comp_sizes_sorted))} by size:")
    for i, size in enumerate(comp_sizes_sorted[:top_n], start=1):
        pct = 100 * size / total_nodes
        print(f"  {label[:-1]} {i}: {size} nodes ({pct:.2f}%)")

    # Optionally show how many nodes are in the largest component
    largest = comp_sizes_sorted[0]
    largest_pct = 100 * largest / total_nodes
    print(f"Largest {label[:-1]} contains {largest} nodes ({largest_pct:.2f}% of all nodes)\n")

# --- 2. Connected components (treat graph as undirected) ---
G_undirected = G.to_undirected()
cc = list(nx.connected_components(G_undirected))
summarize_components(cc, "Connected components", num_nodes)

# --- 3. Strongly connected components (directed) ---
scc = list(nx.strongly_connected_components(G))
summarize_components(scc, "Strongly connected components", num_nodes)

# --- 4. Weakly connected components (directed, connectivity ignoring direction) ---
wcc = list(nx.weakly_connected_components(G))
summarize_components(wcc, "Weakly connected components", num_nodes)

# =========================================
# 1) Clustering coefficient (ignore direction → use undirected graph)
G_undirected = G.to_undirected()
clustering_dict = nx.clustering(G_undirected)  # dict: node -> clustering coeff

# 2) In-degree and out-degree centrality (normalized in [0, 1])
in_deg_c_dict = nx.in_degree_centrality(G)
out_deg_c_dict = nx.out_degree_centrality(G)

# 3) Betweenness centrality (normalized)
betweenness_norm_dict = nx.betweenness_centrality(
    G,
    normalized=True,
    weight=None
)

# 4) Betweenness centrality (non-normalized)
betweenness_raw_dict = nx.betweenness_centrality(
    G,
    normalized=False,
    weight=None
)

# --- Combine into one DataFrame ---
nodes = list(G.nodes())

metrics_df = pd.DataFrame({
    "station_id": nodes,
    "clustering_coeff": [clustering_dict[n] for n in nodes],
    "in_degree_centrality": [in_deg_c_dict[n] for n in nodes],
    "out_degree_centrality": [out_deg_c_dict[n] for n in nodes],
    "betweenness_centrality_norm": [betweenness_norm_dict[n] for n in nodes],
    "betweenness_centrality_raw": [betweenness_raw_dict[n] for n in nodes],
})

# Optional: sort by one of the metrics, e.g. normalized betweenness
metrics_df_sorted = metrics_df.sort_values(
    "betweenness_centrality_norm",
    ascending=False
)

print(metrics_df_sorted.head(5))
#=====================================
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(metrics_df["out_degree_centrality"], bins=20, kde=True, color="skyblue")
plt.title("Out-Degree Centrality Distribution")
plt.xlabel("Out-Degree Centrality")
plt.ylabel("Number of Stations")
plt.show()
#====================================================
# hw 2

# Convert to undirected (standard for clustering coefficient)
G_undirected = G.to_undirected()

# Global clustering coefficient (a.k.a. transitivity)
global_clustering = nx.transitivity(G_undirected)

# Average clustering coefficient (average over all nodes)
avg_clustering = nx.average_clustering(G_undirected)

print(f"Global Clustering Coefficient (Transitivity): {global_clustering:.4f}")
print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
