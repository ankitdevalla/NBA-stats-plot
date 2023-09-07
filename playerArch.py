import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import mplcursors

# Load and preprocess your data
data = pd.read_csv('nba_selected_df_filtered.csv')
features = ["MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PTS",
    "PLUS_MINUS"]
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
# pca = PCA(n_components=len(features))  # Choose the number of components
# X_pca = pca.fit_transform(X_scaled)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Choose the number of components
X_pca = pca.fit_transform(X_scaled)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Create a scatter plot of t-SNE clusters
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data['PLUS_MINUS'])
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Clusters')

# # Create a scatter plot of the PCA clusters
# scatter = plt.scatter(X_pca[:, 0], X_pca[:, len(features) - 1], c=data['PTS']) 
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA Clusters')

# # Apply t-SNE
# tsne = TSNE(n_components=2, random_state=42)  # Choose the number of components
# X_tsne = tsne.fit_transform(X_scaled)

# # Create a scatter plot of the t-SNE clusters
# scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data['PTS'])
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.title('t-SNE Clusters')

# Use mplcursors to add labels when hovering over points
cursor = mplcursors.cursor(scatter, hover=True)

@cursor.connect("add")
def on_add(sel):
    ind = sel.index
    player_name = data.iloc[ind]['PLAYER_NAME']
    sel.annotation.set_text(player_name)

plt.show()


