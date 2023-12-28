import pandas as pd
import plotly.graph_objs as go

# Load data from CSV file
excel_file_path = r'C:\Users\harshini\Desktop\ChatProbe\cluster_5.xlsx'  
text_column_name = 'reasons'  
df = pd.read_excel(excel_file_path)  # Replace 'your_data.csv' with your CSV file path


# Define cluster labels and color mappings
cluster_labels = {
    0: "(Aerolink) Transactions and Balance",
    1: "User enquiries -not enough info available",
    2: "Unresolved Issues personalized Support",
    3: "Browsing and internet connectivity",
    4: "Call related issues - call quality, international roaming"
}

# Define color mapping for clusters
color_map = {
    '(Aerolink) Transactions and Balance': 'rgb(119, 221, 119)',
    'User enquiries -not enough info available': 'rgb(0, 128, 128)',
    'Unresolved Issues personalized Support': 'rgb(255, 153, 0)',
    'Browsing and internet connectivity': 'rgb(0, 191, 255)',
    'Call related issues - call quality, international roaming': 'rgb(255,255,0)'
}

# Create hover text with cluster label and reason information for each point
hover_text = df.apply(
    lambda row: f"Cluster: {cluster_labels.get(row['cluster'], 'Unknown')}<br>Reasons: {row['reasons']}",
    axis=1).tolist()

# Plot using Plotly (3D Scatter Plot)
data = []
for cluster_id, label in cluster_labels.items():
    cluster_data = df[df['cluster'] == cluster_id]
    data.append(go.Scatter3d(
        x=cluster_data['umap_x'],
        y=cluster_data['umap_y'],
        z=cluster_data['umap_z'],
        mode='markers',
        marker=dict(size=5, color=color_map[label], opacity=0.7),
        name=label,
        text=hover_text,
        hoverinfo='text'
    ))

layout = go.Layout(
    title='KMeans Clustering',
    scene=dict(
        xaxis=dict(title='UMAP X'),
        yaxis=dict(title='UMAP Y'),
        zaxis=dict(title='UMAP Z'),
        bgcolor="#FDF7F0",  # Set background color
    ),
)

fig = go.Figure(data=data, layout=layout)
fig.show()
