import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output



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

# Perform PCA for dimensionality reduction (reduce to 3 components)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(embeddings)

# Create a DataFrame with PCA results
df_pca = pd.DataFrame(data=pca_result, columns=['x', 'y', 'z'])

# Assign PCA dimensions to the DataFrame
df['x'] = df_pca['x']
df['y'] = df_pca['y']
df['z'] = df_pca['z']

# Existing code...

# Assign PCA dimensions to the DataFrame
df['x'] = df_pca['x']
df['y'] = df_pca['y']
df['z'] = df_pca['z']

# Load your data here
# Assuming df is your DataFrame containing the necessary data
# Define cluster labels and corresponding colors
cluster_labels = {
    0: "Payment Issues on App",
    1: "Data Plan Change Requests",
    2: "Transaction Related Issues",
    3: "Internet Speed Related Issues",
    4: "Call Related Issues"
    #5: "xyz"
}
 
# Define color mapping for clusters
color_map = {
    'Payment Issues on App': 'rgb(210,222,50)',    #green
    'Data Plan Change Requests': 'rgb(0, 128, 128)',    
    'Transaction Related Issues': 'rgb(242, 140, 40)',    # Orange
    'Internet Speed Related Issues': 'rgb(137, 207, 240)'  ,#blue
    'Call Related Issues':'rgb(255, 0, 0)'#red
    
}

# Define unique intents and corresponding colors
unique_intents = df['Intent'].unique()
intent_colors = {intent: color_map[cluster_labels[i]] for i, intent in enumerate(unique_intents)}

# Create Dash app
app = dash.Dash(__name__)

# Define dropdown options
intent_options = [{'label': intent, 'value': intent} for intent in unique_intents]
# Layout of the app with adjusted positions and styles
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='intent-dropdown',
            options=intent_options,
            value=unique_intents[0],  # Set initial value to the first intent
            clearable=False,
            style={'width': '200px'}  # Set the width of the dropdown
        ),
    ], style={'textAlign': 'center', 'marginTop': '20px'}),  # Center the dropdown with some top margin

    html.Div([
        dcc.Graph(id='pca-kmeans-3d-scatter'),
    ], style={'textAlign': 'center', 'margin': 'auto', 'width': '80%', 'height': '70vh'}),  # Center the graph with specified width and height

    html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        style={
                                            'width': '10px',
                                            'height': '10px',
                                            'background-color': intent_colors[intent],
                                            'display': 'inline-block',
                                            'margin-right': '5px'
                                        }
                                    ),
                                    html.Div(intent, style={'display': 'inline-block'})
                                ],
                                style={'display': 'flex', 'align-items': 'center'}
                            )
                        ]
                    )
                    for intent in unique_intents
                ],
                style={'display': 'flex', 'justify-content': 'center'}
            )
        ],
        style={'display': 'flex', 'justify-content': 'center', 'padding-top': '20px'}  # Add padding at the top of the legend row
    )
])

# Callback to update scatter plot based on selected intent
@app.callback(
    Output('pca-kmeans-3d-scatter', 'figure'),
    [Input('intent-dropdown', 'value')]
)
def update_scatter(intent):
    filtered_df = df[df['Intent'] == intent]

    hover_text = filtered_df.apply(
        lambda row: f"Intent: {row['Intent']}<br>Label: {cluster_labels.get(row['cluster'], 'Unknown')}<br>Cluster: {row['cluster']}<br>Reasons: {row['reasons']}",
        axis=1
    ).tolist()

    scatter = go.Scatter3d(
        x=filtered_df['x'],
        y=filtered_df['y'],
        z=filtered_df['z'],
        mode='markers',
        marker=dict(
            size=8,
            color=intent_colors[intent],  # Use color assigned to the intent
            opacity=0.7,
        ),
        text=hover_text,
        hoverinfo='text',
        hovertemplate='%{text}<extra></extra>',
    )

    layout = go.Layout(
        title='PCA and KMeans Clustering in 3D',
        scene=dict(
            # xaxis=dict(title='Representative dimension 1'),
            # yaxis=dict(title='Representative dimension 2'),
            # zaxis=dict(title='Representative dimension 3'),
            camera=dict(eye=dict(x=0.6584935546502723, y=-2.1226840653466983, z=1.3221567975120296)),
            dragmode='orbit',
            xaxis=dict(showticklabels=False),  # Hide x-axis tick labels
            yaxis=dict(showticklabels=False),  # Hide y-axis tick labels
            zaxis=dict(showticklabels=False),  # Hide z-axis tick labels
        ),
        margin=dict(l=30, r=0, b=0, t=30)
    )

    fig = go.Figure(data=[scatter], layout=layout)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port='2000')