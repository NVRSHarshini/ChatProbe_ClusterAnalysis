import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import dash
from dash import dcc, html
from ast import literal_eval

# Load embeddings from CSV
csv_file_path = r'C:\Users\harshini\Desktop\ChatProbe\Cluster_withOutcomeandEmbedding_5.csv'
df = pd.read_csv(csv_file_path, encoding='utf-8')

# Drop rows with NaN values in 'embeddings' column
df.dropna(subset=['embeddings'], inplace=True)

# Convert string representations of embeddings to lists
df['embeddings'] = df['embeddings'].apply(lambda x: literal_eval(x) if pd.notnull(x) else [])

# Convert embeddings to NumPy arrays
embeddings = np.array(df['embeddings'].tolist())

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
    0: "Payment Issues on App",
    1: "Data Plan Change Requests",
    2: "Transaction Related Issues",
    3: "Internet Speed Related Issues",
    4: "Call Related Issues"
}

# Define color mapping for clusters
color_map = {
    'Payment Issues on App': 'rgb(210,222,50)',    #green
    'Data Plan Change Requests': 'rgb(0, 128, 128)',    
    'Transaction Related Issues': 'rgb(242, 140, 40)',    # Orange
    'Internet Speed Related Issues': 'rgb(137, 207, 240)',  #blue
    'Call Related Issues': 'rgb(255, 0, 0)'  #red
}

# Map cluster labels to colors
df['color'] = df['cluster'].map(cluster_labels).map(color_map)

# Create hover text with cluster label and color information for each point
hover_text = df.apply(lambda row: f"Cluster: {cluster_labels.get(row['cluster'])}<br>Reasons: {row['reasons']}", axis=1).tolist()

# Plot using Plotly (3D Scatter Plot)
scatter = go.Scatter3d(
    x=df['x'],
    y=df['y'],
    z=df['z'],
    mode='markers',
    marker=dict(
        size=7,
        color=df['color'], 
    ),
    text=hover_text,
    hoverinfo='text',
    hovertemplate='%{text}<extra></extra>',
)

layout = go.Layout(
    title={
        'text': 'Categorizing Chatbot Escalation Reasons',
        'font': {'size': 28}  # Adjust the font size as needed
    },
    scene=dict(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False),
        camera=dict(eye=dict(x=0.6584935546502723, y=-2.1226840653466983, z=1.3221567975120296)),  # Set initial camera orientation
    ),
    margin=dict(l=30, r=0, b=0, t=30),
    hoverlabel=dict(
        font=dict(size=17),
        namelength=-1
    ),
)

fig = go.Figure(data=[scatter], layout=layout)


intent_counts = df[df['Intent'] != 'Other']['Intent'].value_counts()


sorted_intents = intent_counts.sort_values(ascending=False).index.tolist()


if 'Other' in sorted_intents:
    sorted_intents.remove('Other')
    sorted_intents.append('Other')

sorted_intents.insert(0, 'All')
intent_options = [{'label': intent, 'value': intent} for intent in sorted_intents]

legend_items = [
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
                                    'background-color': color_map[cluster_labels[i]],
                                    'display': 'inline-block',
                                    'margin-right': '5px',
                                    'margin-left': '25px'  
                                }
                            ),
                            html.Div(cluster_labels[i], style={'display': 'inline-block','font-size':'20px'})
                        ],
                        style={'display': 'flex', 'align-items': 'center'}
                    )
                ]
            )
            for i in range(len(cluster_labels))
        ],
        style={'display': 'flex', 'justify-content': 'center'}
    )
]

x_range_all = [df['x'].min() - 0.1, df['x'].max() + 0.1]  
y_range_all = [df['y'].min() - 0.1, df['y'].max() + 0.1]  
z_range_all = [df['z'].min() - 0.1, df['z'].max() + 0.1]  

# Initialize your Dash app
app = dash.Dash(__name__)

# Define your app layout
app.layout = html.Div([
    html.Div([
        html.Label('Select:', style={'margin-right': '10px', 'color': 'white', 'font-weight': 'bold'}),
        dcc.Dropdown(
            id='intent-dropdown',
            options=intent_options,
            value='All',
            clearable=False,
            style={'width': '200px'}
        ),
    ], style={'position':'relative', 'margin-top': '20px', 'margin-right': '20px'}),

    html.Div([
        dcc.Graph(
            id='pca-kmeans-3d-scatter',
            style={'height': '70vh', 'width': '80vw'}
        )
    ]),
    
    html.Div(legend_items, style={'display': 'flex','padding-left':'120px', 'padding-top': '20px'})
])
# Define scatter plot outside callback with empty data
scatter = go.Scatter3d(
    x=[],
    y=[],
    z=[],
    mode='markers',
    marker=dict(size=7),
    text=[],
    hoverinfo='text',
    hovertemplate='%{text}<extra></extra>',
)

# Callback to update the scatter plot based on selected intent
@app.callback(
    dash.dependencies.Output('pca-kmeans-3d-scatter', 'figure'),
    [dash.dependencies.Input('intent-dropdown', 'value')]
)
def update_scatter(selected_intent):
    filtered_df = df if selected_intent == 'All' else df[df['Intent'] == selected_intent]

    hover_text = filtered_df.apply(
        lambda row: f"Intent: {row['Intent']}<br>Label: {cluster_labels.get(row['cluster'], 'Unknown')}<br>Reasons: {row['reasons']}",
        axis=1
    ).tolist()

    # Update scatter trace data
    scatter.x = filtered_df['x']
    scatter.y = filtered_df['y']
    scatter.z = filtered_df['z']
    scatter.marker.color = filtered_df['color']
    scatter.text = hover_text

    layout = go.Layout(
        title=dict(
        text='Categorizing Of Chatbot Escalation Reasons',
        x=0.5,
         # Set the title position to the center
    ),
        scene=dict(
              xaxis=dict(
            gridcolor='grey',
            showticklabels=False,
            range=x_range_all,
            linewidth=2,
            mirror=True,
            showbackground=False,
            linecolor='black',
            titlefont=dict(color='white')
        ),
        yaxis=dict(
            gridcolor='grey',
            showticklabels=False,
            range=y_range_all,
            linewidth=2,
            mirror=True,
            showbackground=False,
            linecolor='black',
            titlefont=dict(color='white')
        ),
        zaxis=dict(
            gridcolor='grey',
            showticklabels=False,
            range=z_range_all,
            linewidth=2,
            mirror=True,
            showbackground=False,
            linecolor='black',
            titlefont=dict(color='white')
        ),
           
            camera=dict(eye=dict(x=0.6584935546502723, y=-2.1226840653466983, z=1.3221567975120296)),
            aspectmode='manual',  
            aspectratio=dict(x=1, y=1, z=1)  
        ),
        margin=dict(l=30, r=0, b=0, t=30),
        hoverlabel=dict(
            font=dict(size=17),
            namelength=-1
        ),
    )

    fig = go.Figure(data=[scatter], layout=layout)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True,port='1005')