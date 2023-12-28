import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import dash
from dash import dcc, html
from ast import literal_eval
 
# Load embeddings from CSV
csv_file_path = r'C:\Users\harshini\Desktop\ChatProbe\clustering_final_with_embeddings.csv'
df = pd.read_csv(csv_file_path, encoding='utf-8')
 
# Drop rows with NaN values in 'embeddings' column
df.dropna(subset=['embeddings'], inplace=True)
 
# Convert string representations of embeddings to lists
df['embeddings'] = df['embeddings'].apply(lambda x: literal_eval(x) if pd.notnull(x) else [])
 
# Convert embeddings to NumPy arrays
embeddings = np.array(df['embeddings'].tolist())
 
# Perform KMeans clustering
num_clusters = 5  # Define the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df['cluster'] = kmeans.fit_predict(embeddings)
 # Specify columns to keep (original columns + 'cluster')
columns_to_keep = ['Session ID','Intent','Subintent','Outcome','reasons'] 
columns_to_keep.append('cluster')  # Add the 'cluster' column

# Save selected columns with cluster labels into a CSV file
output_csv_path = r'C:\Users\harshini\Desktop\ChatProbe\Cluster_withOutcome_5.csv'
df[columns_to_keep].to_csv(output_csv_path, index=False)
# Perform PCA for dimensionality reduction (reduce to 3 components)


#.........final file......
columns_to_keep = ['Session ID','Intent','Subintent','Outcome','reasons','embeddings'] 
columns_to_keep.append('cluster')  # Add the 'cluster' column

# Save selected columns with cluster labels into a CSV file
output_csv_path = r'C:\Users\harshini\Desktop\ChatProbe\Cluster_withOutcomeandEmbedding_5.csv'
df[columns_to_keep].to_csv(output_csv_path, index=False)


pca = PCA(n_components=3)
pca_result = pca.fit_transform(embeddings)
 
 
# Create a DataFrame with PCA results
df_pca = pd.DataFrame(data=pca_result, columns=['x', 'y', 'z'])
 
# Assign PCA dimensions to the DataFrame
df['x'] = df_pca['x']
df['y'] = df_pca['y']
df['z'] = df_pca['z']
 
# Define cluster labels and corresponding colors
cluster_labels = {
    0: "Transaction, technical Support",
    1: "Balance, Plan Customizations",
    2: "Unresolved Issues personalized Support",
    3: "Connectivity and Service Problems",
    4: "Call issues"
    #5: "xyz"
}
 
# Define color mapping for clusters
color_map = {
    'Transaction, technical Support': 'rgb(119, 221, 119)',    
    'Balance, Plan Customizations': 'rgb(0, 128, 128)',    
    'Unresolved Issues personalized Support': 'rgb(255, 153, 0)',    # Orange
    'Connectivity and Service Problems': 'rgb(0, 191, 255)'  ,
    'Call issues':'rgb(255, 0, 0)'
    #'xyz':'rgb(0,0,0)'
}
 
# Map cluster labels to colors
df['color'] = df['cluster'].map(cluster_labels).map(color_map)
 
# Create hover text with cluster label and color information for each point
hover_text = df.apply(lambda row: f"Cluster: {cluster_labels.get(row['cluster'], 'Unknown')}<br>Reasons: {row['reasons']}", axis=1).tolist()
 
# Plot using Plotly (3D Scatter Plot)
scatter = go.Scatter3d(
    x=df['x'],
    y=df['y'],
    z=df['z'],
    mode='markers',
    marker=dict(
        size=8,
        color=df['color'],  # Use the 'color' column that contains actual colors
        opacity=0.7,
    ),
    text=hover_text,
    hoverinfo='text',
    hovertemplate='%{text}<extra></extra>',
)
 
layout = go.Layout(
    title='PCA and KMeans Clustering in 3D',
    scene=dict(
        xaxis=dict(title='Representative dimension 1'),
        yaxis=dict(title='Representative dimension 2'),
        zaxis=dict(title='Representative dimension 3'),
        camera=dict(eye=dict(x=0.6584935546502723, y=-2.1226840653466983, z=1.3221567975120296)),  # Set initial camera orientation
        #"x": 0.6584935546502723, "y": -2.1226840653466983, "z": 1.3221567975120296 
        dragmode='orbit',
        #bgcolor="#FDF7F0",
    ),
    margin=dict(l=30, r=0, b=0, t=30)
)
 
fig = go.Figure(data=[scatter], layout=layout)
 
# Define legend with cluster labels and colors
legend = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(style={'width': '10px', 'height': '10px', 'background-color': color_map[cluster_labels[i]], 'display': 'inline-block'}),
                        html.Div(cluster_labels[i], style={'display': 'inline-block', 'margin-left': '5px'})
                    ],
                    style={'display': 'flex', 'align-items': 'center'}
                )
                for i in range(len(cluster_labels))
            ],
            style={'text-align': 'right'}
        )
    ],
    style={'position': 'absolute', 'top': '10px', 'right': '10px', 'background-color': 'white', 'padding': '10px'}
)


# Create a Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    dcc.Graph(
        id='pca-kmeans-3d-scatter',
        figure=fig,
        style={'height': '70vh', 'width': '80vw'}
    ),
    legend,  # Add legend to the layout
     
])



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port='2000')
# # Create a Dash app
# app = dash.Dash(__name__)
 
# # Layout of the app
# app.layout = html.Div([
#     dcc.Graph(
#         id='pca-kmeans-3d-scatter',
#         figure=fig,
#         style={'height': '100vh', 'width': '100vw'}
#     ),
#     legend  # Add legend to the layout
# ])
 
# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True,port='2000')