# from selectors import EpollSelector
import flask
import dash
import pandas as pd
 
from dash import Dash,dcc,html,dash_table
import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Output,State,Input
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import ALL
#import umap
 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objs as gobs
import pickle
import base64
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import xmltodict
import numpy as np
import os, time
#import ocrmypdf
import base64
from io import BytesIO
import dash
import json
import requests
import os
import psycopg2 
import re
import configparser
import psycopg2
import requests
import pandas as pd
import pymongo
import os
# import nltk
# from transformers import pipeline
import numpy as np
import ast 
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import dash
from dash import dcc, html
from ast import literal_eval
from dash import dash_table
from dash_table import DataTable

#import requests
external_stylesheets = ['https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css',
# 'https://codepen.io/chriddyp/pen/bWLwgP.css',
                        "https://use.fontawesome.com/releases/v5.10.2/css/all.css",
                        "https://code.ionicframework.com/ionicons/2.0.1/css/ionicons.min.css",
                        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
                        "https://use.fontawesome.com/releases/v5.5.0/css/all.css",
                        "https://use.fontawesome.com/releases/v5.8.1/css/all.css",
                        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
                        "https://fonts.googleapis.com/css2?family=Ubuntu:ital,wght@0,300;0,400;0,500;0,700;1,300;1,400;1,500;1,700&display=swap",
                        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
                       ]


external_scripts = ['https://code.jquery.com/jquery-3.2.1.slim.min.js',
                    'https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js',
                    'https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js',
                    "https://codepen.io/chriddyp/pen/bWLwgP.css",external_stylesheets
                    ]


# Server definition
server = flask.Flask(__name__)
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts,
                server=server,
                suppress_callback_exceptions=True)
# Load the data
chatprobe_DA ='' 
#pd.read_csv('C:\Users\harshini\Desktop\ChatProbe\diagnostic_analysis_FINAL.csv')
file_path = './assets/forecast_results_allmodels.xlsx'

def outcome_dist_chart():


    # Count the occurrences of each unique value in the specified column
    outcome_value_counts = chatprobe_DA['Outcome'].value_counts().reset_index()

    # Create a custom color scale from green to dark green
    color_scale = px.colors.sequential.YlGn[::-2]

    # Create Pie chart figure
    fig = go.Figure(data=[go.Pie(labels=outcome_value_counts['index'], values=outcome_value_counts['Outcome'], marker=dict(colors=color_scale))])

    # Customize figure layout
    fig.update_layout(
        title='Distribution of Conversation Outcomes',
        height=500,  # Set the height of the figure
        width=700,   # Set the width of the figure
        title_x=0.4,
    )
    return fig



desired_outcomes = ["Raised a support ticket", "Connecting to an agent"]
DA_filtered_agent_ticket ='' 
#chatprobe_DA[chatprobe_DA["Outcome"].isin(desired_outcomes)]

def top_intent_dist_chart():

    # Get the top intents in unresolved conversations
    top_intents = DA_filtered_agent_ticket['Intent'].value_counts().reset_index()
    print(top_intents)

    # Create a bar graph using Plotly Express
    fig2 = px.bar(top_intents, y=top_intents['index'], x=top_intents['Intent'],labels={"Intent": 'Number of Conversations', 'index': 'Intent'}, title='Top Intents in Unresolved Conversations',  # Set orientation to horizontal
        color='Intent',   # Use 'Intent' as the color dimension
        color_discrete_sequence=px.colors.qualitative.Set1  # Set custom colors
    )

    # Customize figure layout to center the title
    fig2.update_layout(
        title_x=0.5,  # Set the title to the center of the figure horizontally
    )
    return fig2

def intent_dist_overweek_chart():

    # Convert 'Timestamp' to datetime format
    DA_filtered_agent_ticket['Timestamp'] = pd.to_datetime(DA_filtered_agent_ticket['Timestamp'], format="%d-%m-%Y %H:%M")
    
    # Extract the starting day of the week from the date
    DA_filtered_agent_ticket['Week_Start_Day'] = DA_filtered_agent_ticket['Timestamp'].dt.to_period('W-SUN').dt.start_time
    
    # Group by week and intent to get the count
    intent_distribution = DA_filtered_agent_ticket.groupby(['Week_Start_Day', 'Intent']).size().reset_index(name='Count')
    
    # Calculate the total count for each week
    total_count_per_week = intent_distribution.groupby('Week_Start_Day')['Count'].sum()
    
    # Calculate the percentage distribution
    intent_distribution['Percentage'] = (
        intent_distribution['Count'] / intent_distribution['Week_Start_Day'].map(total_count_per_week) * 100
    )
    
    # Create a normalized bar plot using Plotly Express
    fig3 = px.bar(intent_distribution, x='Week_Start_Day', y='Percentage', color='Intent',
                labels={'Percentage': 'Percentage of Conversations', 'Week_Start_Day': 'Weeks (1st Sep 2023 - 31st Oct 2023)'},
                title='Distribution of Intents in unresolved conversations over Week',
                color_discrete_sequence=px.colors.sequential.YlGn[::-2])
    
    # Customize figure layout to center the title
    fig3.update_layout(
        title_x=0.5,  # Set the title to the center of the figure horizontally
    )

    return fig3

def trend_chart():
    DA_filtered_agent_ticket['Timestamp'] = pd.to_datetime(DA_filtered_agent_ticket['Timestamp'], format="%d-%m-%Y %H:%M")
    DA_filtered_agent_ticket['Week_Start_Day'] = DA_filtered_agent_ticket['Timestamp'].dt.to_period('W-SUN').dt.start_time
    unresolved_distribution = DA_filtered_agent_ticket.groupby('Week_Start_Day').size().reset_index(name='Count')
    
    # Create a line plot using Plotly Express
    fig4 = px.line(unresolved_distribution, x='Week_Start_Day', y='Count',
                labels={'Count': 'Number of Unresolved Conversations','Week_Start_Day': 'Weeks (1st Sep 2023 - 31st Oct 2023)'},
                title='Trend of Unresolved Conversations Over Weeks')
    
    
    # Customize figure layout to center the title
    fig4.update_layout(
        title_x=0.5,  # Set the title to the center of the figure horizontally
    )
    return fig4

# Sample data
data = {
    'cluster labels': [0, 1, 2, 3,4],
    'Labels and Themes': ['Transaction and Technical Support', 'Balance, Plan Customization', 'Unresolved Issues personalized Support','Connectivity and Service Problems','Call issues'],
    'Connecting to an agent': [7, 8, 11, 6,10],
    'Raised a Support Ticket': [2,1,7,4,7],
    'Grand Total': [9,9,19,10,17]
}

# Sample data
sample_data = {
    'Intent': ['Network-related issue', 'Network-related issue', 'Network-related issue'],
    'Subintent': [
        ['Issue with calls', 'Audio quality (echo)'],
        ['Internet issue', 'Streaming video'],
        ['5G issue', 'Check handset 5G ready']
    ],
    'Outcome': ['Resolved by bot', 'Resolved by bot', 'Resolved by bot'],
    'Reason_not_resolved': ['N/A', 'N/A', 'N/A']
}

def load_data():
    df = pd.read_excel(file_path)
    return df

def clustering_analysis():  
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
            size=5,
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
            xaxis=dict(title='Representative dimension 1', gridcolor='black'),
            yaxis=dict(title='Representative dimension 2', gridcolor='black'),
            zaxis=dict(title='Representative dimension 3', gridcolor='black'),
                bgcolor='white',
            camera=dict(eye=dict(x=0.6584935546502723, y=-2.1226840653466983, z=1.3221567975120296)),  # Set initial camera orientation
            #"x": 0.6584935546502723, "y": -2.1226840653466983, "z": 1.3221567975120296
            #dragmode='orbit',
            #bgcolor="#FDF7F0",
        ), 
        margin=dict(l=30, r=0, b=0, t=30),
        hoverlabel=dict(
        font=dict(size=17),  
        namelength=-1
    )
    )
    
    fig = go.Figure(data=[scatter], layout=layout)

    return fig

header = dbc.Navbar(
    html.Div(
        [
            html.Div(
                [
                    # html.A(html.Img(src=app.get_asset_url("exflogo.png"), className='brand-image', style={'margin': '-8px -7px','background': '#fff','height':'50px'})),
                    # html.A(html.Img(src=app.get_asset_url("veolia_logo.png"), className='brand-image', style={'margin': '0px -8px 0px 0px','height': '4rem'})),

                ],
                className='float-left'
            ),
            html.Div([
           
            html.Div(
                [
                    html.Div(
                [
            # html.A(html.Img(src=app.get_asset_url("doc_logo.png"), className='title-image',
            #                style={'height': '4rem', 'margin-top':'5px', 'align':'right', 'margin-Left':'0px', 'padding-left':'10px', 'padding-bottom':'3px'})),
                html.Span('Chat', style={'text-shadow':'2px 2px #2b272726','color':'white', 'font-weight':'500', 'padding-left':'14px', 'textAlign': 'left','font-size':'34px'}),
                html.Span('Probe : Shaping Superior Chatbot  Experiences', style={'color':'white','font-size':'34px','text-shadow':'2px 2px #2b272726','font-weight': '500', 'padding-left':'0px', 'textAlign': 'left'}),
                html.Br()
                ],
                className=' text-center', 
                ), 
                ],style={})],style={}),
            html.Div(
                [
                    # html.A(html.Img(src=app.get_asset_url("veolia_logo.png"), className='brand-image', style={'margin': '0px -8px 0px 0px','height': '4rem'})),
                ],
                className='float-right', style={
                    # 'padding-left':'5%'
                    }
            ), 
        ],
        className='container-fluid',
        style={'width': '100%','padding-right': '7.5px','padding-left': '7.5px','background':'#DE8F5F'}
    ),
    color="white",
    style={
        'padding':'0px'
        # 'background-image': 'linear-gradient(to right, rgb(231 143 101), rgb(231 143 101))'
        }
)
points = [
    "Chatbots have become ubiquitos in our digital landscape, serving as virtual assistants in a wide array of contexts. Designing these chatbots is inherently an iteravtive process, where the feedback from users and actual interaction conversations with the bot are highly helpful",
    "This application demonstrates analyzing Chatbot conversations to determine why specific user inquries are not adequately addressed by the chatbot, leading to their transfer to live agents or conversion into support tickets",
    "100 bot conversations were analyzed and structured information summarized and extracted. This was further used for visualization.",
]
data_summary1 = dbc.Card(
            dbc.CardBody(
                html.Ul([html.Li(point) for point in points],style={'font-size': '18px'}),className=" datasummary",style={' background': '#F5EEC8 !importance','width': '96%'}
            ),
            className="mt-3 ",style={'border':'none','align-items': 'center'}
        )
# data_summary=html.Div([
#     dbc.Card([
#         dbc.Row(
#         dbc.Col(
#             html.H3("Maximizing chatbot effectiveness by leveraging AI to analyze, visualize, and refine customer interactions, reducing the need for live support escalation",
#                     style={'text-align': 'center','font-size': '15px'}),
#             width={'size': 6, 'offset': 3}
#         )
#     ),
#         dbc.Row([
#             dbc.Col(
#                 dbc.Card([
#                     html.Img(src="./assets/extract.png" , style={'width': '100%', 'height': 'auto','margin-top':'15px'}),
#                     dbc.CardBody([
#                         # html.H4("Card 1 Content", className="card-title"),
#                         html.Ul([
#                             html.Li("•  Identify user intent and chat outcome ", style={'display': 'inline-block'}),
#                             html.Li("• Summarize reasons for user transfer", style={'display': 'inline-block'}),
#                             html.Li("• Capture structured conversation data", style={'display': 'inline-block'}),
#                             # Add more bullet points as needed
#                         ], style={'list-style-type': 'none' ,'display': 'grid'})
#                     ],style={'padding':'35px'}),
#                 ], className="mb-3 ml-3",style={'box-shadow': 'rgb(221, 221, 221) 1px 1px 4px','height':'20rem'})
#             ),
#              dbc.Card([
#                  dbc.Card(html.H4("Analyze", className="mb-2 card-title text-center"),className="mt-2",style={'width':'94%','background': '#B3A492'}),
#                   dbc.Row([
#             dbc.Col(
#                 dbc.Card([
#                     html.Img(src="./assets/visualize.png" , style={'width': '100%', 'height': 'auto','margin-top':'15px'}),
#                     dbc.CardBody([
#                         # html.H4("Card 1 Content", className="card-title"),
#                         html.Ul([
#                             html.Li("• Chart user intents and outcomes", style={'display': 'inline-block'}),
#                             html.Li("• Display summaries ", style={'display': 'inline-block'}),
#                             html.Li("• Use data-driven visuals for clarity ", style={'display': 'inline-block'}),
#                             # Add more bullet points as needed
#                         ], style={'list-style-type': 'none' ,'display': 'grid'})
#                     ],style={'padding':'10px'}),
#                 ], className="mb-3 mt-3 ml-3",style={'height':'15rem','box-shadow': 'rgb(221, 221, 221) 1px 1px 4px'})
#             ),
#             dbc.Col(
#                 dbc.Card([
#                     html.Img(src="./assets/cluster.png" , style={'width': '100%', 'height': 'auto','margin-top':'15px'}),
#                     dbc.CardBody([
#                         # html.H4("Card 1 Content", className="card-title"),
#                         html.Ul([
#                             html.Li("• Perform semantic analysis on reasons", style={'display': 'inline-block'}),
#                             html.Li("• Cluster similar issues", style={'display': 'inline-block'}),
#                             html.Li("• Identify common escalation triggers", style={'display': 'inline-block'}),
#                             # Add more bullet points as needed
#                         ], style={'list-style-type': 'none' ,'display': 'grid'})
#                     ],style={'padding':'10px'}),
#                 ], className="mb-3 mt-3 mr-3",style={'height':'15rem','box-shadow': 'rgb(221, 221, 221) 1px 1px 4px'})
#             )
#             ]),],style={'height':'20rem','width':'45%','align-items': 'center','box-shadow': 'rgb(221, 221, 221) 1px 1px 4px',    'margin-bottom': '17px'}),
#             dbc.Col(
#                 dbc.Card([
#                     html.Img(src="./assets/optimize.png" , style={'width': '100%', 'height': 'auto','margin-top':'15px'}),
#                     dbc.CardBody([
#                         # html.H4("Card 3 Content", className="card-title"),
#                         html.Ul([
#                             html.Li("• Enhance user’s bot experience", style={'display': 'inline-block'}),
#                             html.Li("• Minimize live agent transfers &  tickets", style={'display': 'inline-block'}),
#                             html.Li("• Identify Process optimization opportunities", style={'display': 'inline-block'}),
#                             # Add more bullet points as needed
#                         ], style={'list-style-type': 'none' ,'display': 'grid'})
#                     ],style={'padding':'36px'}),
#                 ], className="mb-3 mr-3",style={'height':'20rem','box-shadow': 'rgb(221, 221, 221) 1px 1px 4px'})
#             ),
#         ])], style={'width': '96%'})
# ], style={'justify-content': 'center', 'display': 'flex'})
data_summary=html.Div([
    dbc.Card([
        dbc.Row(
    dbc.Col(
        html.Div(
        html.H3(
            "Maximizing Chatbot effectiveness by leveraging AI to analyze, visualize, and refine customer interactions, reducing the need for live support escalation",
            style={'text-align': 'center', 'font-size': '20px', 'margin': '23px 23px 23px 23px'},
        ),)
    )
),

        dbc.Row([
            dbc.Col([
                  dbc.Card([
                       dbc.CardBody([
                    html.Div([
                    html.Img(src="./assets/filter.png", style={'width': '50px', 'height': '50px'}),
                    html.Span("Extract", style={'margin-left': '10px', 'vertical-align': 'middle','font-weight':'bold','font-size':'25px'})
                     ], style={'display': 'flex', 'align-items': 'center'}),
                   
                    html.Ul([
                        html.Li("•  Identify user intent and chat outcome ", style={'display': 'inline-block'}),
                        html.Li("• Summarize reasons for user transfer", style={'display': 'inline-block'}),
                        html.Li("• Capture structured conversation data", style={'display': 'inline-block'}),
                    ], style={'list-style-type': 'none', 'display': 'grid','font-size': '18px','margin-top': '12px','margin-left': '5px'})
                ], style={'padding': '10px'}),
                     ], className="mb-3", style={  'box-shadow': 'rgb(221, 221, 221) 1px 1px 4px' , 'margin': '0px 15px 0px 15px'})
                ]),
                html.Div([
                dbc.Row([     
                    dbc.Card(html.H4("Analyze", className="mb-2 card-title text-center"),className=" ",style={'width':'96%','background': '#B3A492'}),
                    ]),
                dbc.Row([ 
                        dbc.Col([
                            dbc.Card([
                            
                                # html.Img(src="./assets/visualize.png", style={'width': '100%', 'height': 'auto', 'margin-top': '15px'}),
                                dbc.CardBody([
                                    html.Div([
                    html.Img(src="./assets/laptop.png", style={'width': '50px', 'height': '50px'}),
                    html.Span("Visualize", style={'margin-left': '10px', 'vertical-align': 'middle','font-weight':'bold','font-size':'25px'})
                ], style={'display': 'flex', 'align-items': 'center'}),
                                    html.Ul([
                                        html.Li("• Chart user intents and outcomes", style={'display': 'inline-block'}),
                                        html.Li("• Display summaries ", style={'display': 'inline-block'}),
                                        html.Li("• Use data-driven visuals for clarity ", style={'display': 'inline-block'}),
                                    ], style={'list-style-type': 'none', 'display': 'grid','font-size': '18px','margin-top': '12px','margin-left': '5px'})
                                ], style={'padding': '10px'}),
                            ], className="mb-3", style={  'box-shadow': 'rgb(221, 221, 221) 1px 1px 4px' , 'margin': '15px 0px 15px 0px'})
                        ],style={'padding-left':'0px'}),
                        dbc.Col([
                    dbc.Card([
                        # html.Img(src="./assets/cluster.png", style={'width': '100%', 'height': 'auto', 'margin-top': '15px'}),
                        dbc.CardBody([
                            html.Div([
            html.Img(src="./assets/clustering.png", style={'width': '50px', 'height': '50px'}),
            html.Span("Cluster", style={'margin-left': '10px', 'vertical-align': 'middle','font-weight':'bold','font-size':'25px'})
        ], style={'display': 'flex', 'align-items': 'center'}),
                            html.Ul([
                                html.Li("• Perform semantic analysis on reasons", style={'display': 'inline-block'}),
                                html.Li("• Cluster similar issues", style={'display': 'inline-block'}),
                                html.Li("• Identify common escalation triggers", style={'display': 'inline-block'}),
                            ], style={'list-style-type': 'none', 'display': 'grid','font-size': '18px','margin-top': '12px','margin-left': '5px'})
                        ], style={'padding': '10px'}),
                    ], className="mb-3", style={  'box-shadow': 'rgb(221, 221, 221) 1px 1px 4px' , 'margin': '15px 23px 15px 0px'})
            ],style={'padding-left':'0px'}),
                ]),
                ]),
            dbc.Col(
                dbc.Card([
                    # html.Img(src="./assets/optimize.png", style={'width': '100%', 'height': 'auto', 'margin-top': '15px'}),
                    dbc.CardBody([
                         html.Div([
        html.Img(src="./assets/filter.png", style={'width': '50px', 'height': '50px'}),
        html.Span("Optimize", style={'margin-left': '10px', 'vertical-align': 'middle','font-weight':'bold','font-size':'25px'})
    ], style={'display': 'flex', 'align-items': 'center'}),
                        html.Ul([
                            html.Li("• Enhance user’s bot experience", style={'display': 'inline-block'}),
                            html.Li("• Minimize live agent transfers &  tickets", style={'display': 'inline-block'}),
                            html.Li("• Identify Process optimization opportunities", style={'display': 'inline-block'}),
                        ], style={'list-style-type': 'none', 'display': 'grid','font-size': '18px','margin-top': '12px','margin-left': '5px'})
                    ], style={'padding': '10px'}),
                ], className="mb-3", style={  'box-shadow': 'rgb(221, 221, 221) 1px 1px 4px' , 'margin': '0px 15px 0px 0px'})
            ),


        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'})
    ], style={'width': '96%', 'margin-left': '2%', 'margin-right': '2%', 'margin-top': '20px', 'margin-bottom': '20px'})
], style={'justify-content': 'center', 'display': 'flex'})


class App():
    def __init__(self):
        pass

    def get_summary_content(self):
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
        
        card_content1 = [
                
                    dbc.CardBody(
                        [
                            html.H5("# of conversations", className="card-title",style={'color': 'black'}),
                            html.H5(
                                "100",
                                className="card-text",style={'font-weight': 'bold','color': 'black'}
                            ),
                        ],style={'box-shadow': '1px 1px 4px #dddddd','text-align': 'center'}
                    ),
                ]
        conversation1="""Bot: Hi, we are here to help you. Go ahead, select your issue:
1. Network-related issue
2. Billing and payment issue
3. Balance, Recharge and validity
4. Plan/pack related issues
5. Others
User: 3_Others
Bot: Feel free to share your concerns with me so I can assist you accordingly
User: deactivate hellotune
Bot: You can manage your hellotune by tapping on 'manage hellotune' button below   1.manage hello tune  2.manage later  3. watch a video
User: 3.2_manage_later
Bot: Was that helpful? 1. helpful  2.Not helpful
User: 3.2.1_helpful
Bot: Glad we could help!
"""
        conversation2="""Bot: Hi, we are here to help you. Go ahead, select your issue:
1. Network-related issue
2. Billing and payment issue
3. Balance, Recharge and validity
4. Plan/pack related issues
5. Others
User: 1_network_issue
Bot: You have selected 'Network-related issue'. Please help us identify your issue from the options below:
1. Internet issue
2. 5G issue
3. Issue with calls
4. International roaming
5. My issue is not listed here
User: 1.1_internet_issue
Bot: Are you at the location of issue? 1.Yes  2.No
User: 1.1.1_yes
Bot: Hold on while we fetch your current location.(location).got your location. Select your issue:
1.Are you having trouble loading specific websites or experiencing slow browsing?
2.Is your problem related to streaming videos online? Specify if you're facing buffering, freezing, or playback issues
3.Are you experiencing problems with video calls? Please share details about the specific issues.
4.Any other Queries
User: 1.1.1.3_issue_with_videocalling
Bot: Please share details about the specific issues.
User: frequent call drops
Bot: There are several potential factors that could be contributing to the problem.                                                                                                               1.Check Internet Connection:
Ensure that you have a stable and high-speed internet connection. Video calls require sufficient bandwidth for smooth communication. Connect to a reliable Wi-Fi network or use mobile data with a strong signal.               
2.Close Background Apps.                                                                                  
3.Check Camera and Microphone Permissions:
Make sure the app has the necessary permissions to access your device's camera and microphone. 
User: tried, not solving
Bot: Apologies, Would you like to 1. Raise a ticket 2.Start a live chat
User: 1.1.1.3.2_start_live_chat
Bot: Absolutely! I'll connect you to our customer support team.
 Please hold on for a moment while I transfer you.
"""
        conversation3="""Bot: Hi, we are here to help you. Go ahead, select your issue:
1. Network-related issue
2. Billing and payment issue
3. Balance, Recharge and validity
4. Plan/pack related issues
5. Others
User: 2_Billing_and_payment_issue
Bot: You have selected 'Billing and payment issue'. Please help us identify your issue from the options below:
1.Payment history 
2.Payments bank related
3.Queries related to payment
4.My issue is not listed here
User: 2.2_Payments_bank_related
Bot: For issues related to Aerolink Payments Bank, please tap on the button below:
[1.Aerolink Payments Bank Account]
User: 2.2.1_Aerolink_Payments_Bank_Account
Bot: Please select your issue:
1.Transaction Failures
2.Account Balance Discrepancies
3.Mobile Banking App Errors
User: No, I want information regarding downloading reciepts.
Bot: Feel free to share your concerns with me so I can assist you accordingly.
User: The receipt download links have expired, what to do now?
Bot: Unfortunately, expired download links can't be reactivated .
User: I am unable to take screenshots of the transactions as well!
Bot: I'm unable to understand your trouble. Please provide more details.
User: Since there is no download option tried to take screenshot but there's restriction
Bot: I apologize for the inconvenience.  There might be technical error validating your account.
 Would you like to 1. Raise a ticket 2. Start a live chat
User: No rights for the bot to solve the issue , 
         2.2.1.1_create_ticket
Bot: Your ticket has been created and reference no is 567610XXXX
"""
        card_content2 = [
               
                    dbc.CardBody(
                        [
                            html.H5("Date Range", className="card-title text-center",style={'color': 'black'}),
                            html.H5(
                                "Sep '23-Oct '23",
                                className="card-text",style={'font-weight': 'bold','color': 'black'}
                            ),
                        ],style={'box-shadow': '1px 1px 4px #dddddd','text-align': 'center'}
                    ),
                ]
        card_content3 = [
                 
                    dbc.CardBody(
                        [
                            html.H5("Resolved by bot", className="card-title",style={'color': 'black'}),
                            html.H5(
                                "36%",
                                className="card-text",style={'font-weight': 'bold','color': 'black'}
                            ),
                        ],style={'box-shadow': '1px 1px 4px #dddddd','text-align': 'center'}
                    ),
                ]
        card_content4 = [
                 
                    dbc.CardBody(
                        [
                            html.H5("Transferred to Live Agent", className="card-title",style={'color': 'black'}),
                            html.H5(
                                "39%",
                                className="card-text",style={'font-weight': 'bold','color': 'black'}
                            ),
                        ],style={'box-shadow': '1px 1px 4px #dddddd','text-align': 'center'}
                    ),
                ]
        card_content5 = [
                 
                    dbc.CardBody(
                        [
                            html.H5("Support Ticket", className="card-title",style={'color': 'black'}),
                            html.H5(
                                "21%",
                                className="card-text",style={'font-weight': 'bold','color': 'black'}
                            ),
                        ],style={'box-shadow': '1px 1px 4px #dddddd','text-align': 'center'}
                    ),
                ]
        button1 = [
                 
                    dbc.CardBody(
                        [
                            # html.H5("Support Ticket", className="card-title",style={'color': 'black'}),
                            # html.H5(
                            #     "21%",
                            #     className="card-text",style={'font-weight': 'bold','color': 'black'}
                            # ),
                            dcc.Download(id="download-data1"),
                                            dbc.Button(
                                                id="Sample-Conversation",
                                                color="info",
                                                n_clicks=0,
                                                children=[
                                                    html.I(
                                                        className="bi bi-download"),
                                                    "    Sample Conversation",
                                                ],
                                                style={"max-width": "fit-content",
                                                       "font-family": "sans-serif",
                                                       "margin-right": "0.3rem",
                                                         "background":"#A7D397",
                                                       "border-color":"#A7D397",
                                                       },
                                            ),
                                            dcc.Input(
                                                id="samplefilename1",
                                                type="text",
                                                value="./assets/diagnostic_analysis_FINAL.csv",
                                                style={"display": "none"},
                                            ),
                        ],style={'box-shadow': '1px 1px 4px #dddddd','text-align': 'center'}
                    ),
                ]
        button2 = [
                 
                    dbc.CardBody(
                        [ 
                            dcc.Download(id="download-data2"),
                                            dbc.Button(
                                                id="Extracted-Output",
                                                color="info",
                                                n_clicks=0,
                                                children=[
                                                    html.I(
                                                        className="bi bi-download"),
                                                    "    Extracted Output",
                                                ],
                                                style={"max-width": "fit-content",
                                                       "font-family": "sans-serif",
                                                       "margin-right": "0.3rem",
                                                       "background":"#A7D397",
                                                       "border-color":"#A7D397",
                                                       },
                                            ),
                                            dcc.Input(
                                                id="samplefilename2",
                                                type="text",
                                                value="./assets/diagnostic_analysis_FINAL.csv",
                                                style={"display": "none"},
                                            ),
                        ],style={
                           
                            'text-align': 'center'}
                    ),
                ]
        card_content6 = [
                 
                    dbc.CardBody(
                        [
                            html.H5("Sample Conversation 1", className="card-title",style={'color': 'black','font-weight':'bold'}),
                            dcc.Textarea(
        id='input-textarea1',
        placeholder='Edit Conversation 1...',
        value=conversation1,
        style={'width': '100%', 'height': 150},
    ),
                           
                        ],style={
                           
                            'text-align': 'center','background':'#F4DFB6'}
                    ),
                ]
        card_content7 = [
                 
                    dbc.CardBody(
                        [
                            html.H5("Sample Conversation 2", className="card-title",style={'color': 'black','font-weight':'bold'}),
                               dcc.Textarea(
        id='input-textarea2',
        placeholder='Edit Conversation 2...',
        value=conversation2,
        style={'width': '100%', 'height': 150},
    ),
                            # html.H5(
                            #     "zzz%",
                            #     className="card-text",style={'font-weight': 'bold','color': 'black'}
                            # ),
                        ],style={
                            # 'box-shadow': '1px 1px 4px #dddddd',
                            'text-align': 'center','background':'#F4DFB6'}
                    ),
                ]
        card_content8 = [
                 
                    dbc.CardBody(
                        [
                            html.H5("Sample Conversation 3", className="card-title",style={'color': 'black','font-weight':'bold'}),
                               dcc.Textarea(
        id='input-textarea3',
        placeholder='Edit Conversation 3...',
        value=conversation3,
        style={'width': '100%', 'height': 150},
    ),
                            # html.H5(
                            #     "zzz%",
                            #     className="card-text",style={'font-weight': 'bold','color': 'black'}
                            # ),
                        ],style={
                            # 'box-shadow': '1px 1px 4px #dddddd',
                            'text-align': 'center','background':'#F4DFB6'}
                    ),
                ]
        card_content9 = [
                 
                    html.Button(
                        [
                            html.H5("Extract Structured Data", className="card-title",style={'color': 'white',"font-weight":"bold"}),
                            # html.H5(
                            #     "zzz%",
                            #     className="card-text",style={'font-weight': 'bold','color': 'black'}
                            # ),
                        ],id='ExtractStructuredData',n_clicks=0, className="ExtractStructuredData",style={
                            # 'box-shadow': 'rgb(117 110 110) 1px 1px 4px',
                                                                                                          'text-align': 'center','background':'#9A4444','padding': '10px 0px 0px 0px'}
                    ),
                ]
        card_content10 = [
                 
                    html.Button(
                        [
                            html.H5("View Prompt", className="card-title",style={'color': 'white',"font-weight":"bold"}),
                            # html.H5(
                            #     " ",
                            #     className="card-text",style={'font-weight': 'bold','color': 'black'}
                            # ),
                        ] ,id="toggle_sidebar" ,className="View-Prompt",style={
                            # 'box-shadow': 'rgb(117 110 110) 1px 1px 4px',
                            'text-align': 'center','background':'#9A4444','padding': '10px 0px 0px 0px'}
                    ),
                ]
        card_content11 = [
                    dbc.Card(
                    dbc.CardBody(
                        [
                              dash_table.DataTable(
        id='table',
        columns=[{'name': col, 'id': col} for col in data.keys()],
        data=[{col: data[col][i] for col in data.keys()} for i in range(len(data['cluster labels']))],
        style_header={'backgroundColor': '#89B269', 'fontWeight': 'bold', 'color': '#FFFFFF', 'textAlign': 'center'},
        style_cell={'backgroundColor': '#D3E2C1', 'color': '#000000', 'textAlign': 'center'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#D3E2C1'
            },
            {
                'if': {'row_index': 'even'},
                'backgroundColor': '#A7D397'
            }
        ]
    )
                            # html.H5("View Prompt", className="card-title",style={'color': 'white',"font-weight":"bold"}),
                            # html.H5(
                            #     " ",
                            #     className="card-text",style={'font-weight': 'bold','color': 'black'}
                            # ),
                        ] ,className="View-Prompt",style={'box-shadow': 'rgb(117 110 110) 1px 1px 4px','text-align': 'center',
                                                        #   'background':'#D6C46D',
                                                          'padding': '0px 0px 0px 0px'}
                    ),)
                ]
        tab1_content = dbc.Card(
            dbc.CardBody([
                   dbc.Row(
            [
                dbc.Col(dbc.Card(card_content1, color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'})),
                dbc.Col(
                    dbc.Card(card_content2, color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'})
                ),
                dbc.Col(dbc.Card(card_content3, color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'})),
                 dbc.Col(dbc.Card(card_content4, color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'})),
                  dbc.Col(dbc.Card(card_content5, color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'})),
                    dbc.Col(dbc.Card(button1, color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'})),
                    dbc.Col(dbc.Card(button2, color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'})),
            ],
            className="",
        ),
        
       
#````````````````````````````````````````````
    #    dbc.Card(
    #         dbc.CardBody([
    #     dbc.Row([

    #         dbc.Col(
    #     dcc.Graph(
    #             id='conversation-outcomes1',
    #             figure=outcome_dist_chart()
    #         )
    #     ),
        
    #     dbc.Col(dcc.Graph(
    #             id='conversation-outcomes2',
    #             figure=top_intent_dist_chart()
    #         )

    #     )
        
    #     ]),
    #     dbc.Row([
    #         dbc.Col(
    #     dcc.Graph(
    #             id='conversation-outcomes3',
    #             figure=intent_dist_overweek_chart()
    #         )),
    #          dbc.Col(dcc.Graph(
    #             id='conversation-outcomes4',
    #             figure=trend_chart()
    #         )

    #     )
    #     ]
    #     )
    #         ]),className="mt-3"
    #         )
#````````````````````````````````````````````
#alignmnt of charts
    #                 dbc.Card(
    #         dbc.CardBody([
    #     dbc.Row([

    #         dbc.Col(
    #             dbc.Card(
    #         dbc.CardBody([
    #     dcc.Graph(
    #             id='conversation-outcomes1',
    #             figure=outcome_dist_chart()
    #         )
        
    #         ]))
    #     ),
        
    #     dbc.Col(
    #          dbc.Card(
    #         dbc.CardBody([
    #         dcc.Graph(
    #             id='conversation-outcomes2',
    #             figure=top_intent_dist_chart()
    #         )
    # ]),style={'height': '541px'})
    #     )
        
    #     ]),
    #     dbc.Row([
    #         dbc.Col(
    #              dbc.Card(
    #         dbc.CardBody([
    #     dcc.Graph(
    #             id='conversation-outcomes3',
    #             figure=intent_dist_overweek_chart(),
    #         )   ]),className="mt-3")),
    #          dbc.Col(
    #                 dbc.Card(
    #         dbc.CardBody([
    #              dcc.Graph(
    #             id='conversation-outcomes4',
    #             figure=trend_chart()
    #         )
    #                  ]),className="mt-3")

    #     )
    #     ]
    #     )
    #         ]),className="mt-3"
    #         ),
#````````````````````````````````````````````
dbc.Card(
    dbc.CardBody([
        # dbc.Row([
        #     dbc.Col(
        #         dbc.Card(
        #             dbc.CardBody([
        #                 dcc.Graph(
        #                     id='conversation-outcomes1',
        #                     figure=outcome_dist_chart()
        #                 )
        #             ])
        #         ),
        #         lg=6, md=6, sm=12, xs=12  # Adjust these values based on your layout requirements
        #     ),
        #     dbc.Col(
        #         dbc.Card(
        #             dbc.CardBody([
        #                 dcc.Graph(
        #                     id='conversation-outcomes2',
        #                     figure=top_intent_dist_chart()
        #                 )
        #             ]),style={'height': '541px'}
        #         ),
        #         lg=6, md=6, sm=12, xs=12  # Adjust these values based on your layout requirements
        #     )
        # ]),
        # dbc.Row([
        #     dbc.Col(
        #         dbc.Card(
        #             dbc.CardBody([
        #                 dcc.Graph(
        #                     id='conversation-outcomes3',
        #                     figure=intent_dist_overweek_chart(),
        #                 )
        #             ]),className="mt-3"
        #         ),
        #         lg=6, md=6, sm=12, xs=12  # Adjust these values based on your layout requirements
        #     ),
        #     dbc.Col(
        #         dbc.Card(
        #             dbc.CardBody([
        #                 dcc.Graph(
        #                     id='conversation-outcomes4',
        #                     figure=trend_chart()
        #                 )
        #             ]),className="mt-3"
        #         ),
        #         lg=6, md=6, sm=12, xs=12  # Adjust these values based on your layout requirements
        #     )
        # ]),
         dbc.Row(
            # dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        dcc.Graph(
                            id='conversation-outcomes4',
                            figure=clustering_analysis()
                        ),
                        html.Div(
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
                    ]),className="mt-3",style={"width": "94.8rem"}
                ),style={'justify-content': 'center'}
                # lg=6, md=6, sm=12, xs=12  # Adjust these values based on your layout requirements
            # )
        ), dbc.Row(
            [
                dbc.Col(dbc.Card(card_content11, color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'}),style={'max-width': '100%'}),
             ],
            className="mt-4",style={'justify-content': 'center'}
        ),
        dbc.Row([
    #         dbc.Card(
    #         dcc.Link(
    #     href='https://public.tableau.com/app/profile/exafluence2701/viz/Chatprobe/ChatprobeConversationAnalysis',  # Replace with the actual API endpoint
    # ))
    # dbc.NavbarSimple(
    #     children=[
    #         dbc.NavItem(dbc.NavLink("Tableau Online Dashboard", href="https://public.tableau.com/app/profile/exafluence2701/viz/Chatprobe/ChatprobeConversationAnalysis")),
    #     ],
    #     brand="Click on Dashboard",
    #     brand_href="#",
    #     color="primary",
    #     dark=True,
    # ),
    html.Div([
    # Three div elements in a single row
    # html.Div("click on", style={'display': 'inline-block', 'margin': '10px','font-size': '21px'}),
    # html.Div(html.A(html.Img(
    #     className="mt-4",
    #             src="./assets/tableau.png",  # Replace with the actual path to your image
    #             style={'width': '135px', 'height': '100px','filter': 'drop-shadow(2px 4px 8px hsla(0deg, 0%, 0%, 0.5))'},  # Set the width and height as needed
    #         ), href="https://public.tableau.com/app/profile/exafluence2701/viz/Chatprobe/ChatprobeConversationAnalysis",
    #     target="_blank"  # Open link in a new tab
    # ), style={'display': 'inline-block', 'margin': '10px','filter': 'drop-shadow(2px 4px 8px hsla(0deg, 0%, 0%, 0.5))'}),

    dcc.Link(
    "Link to the Tableau Dashboard",
    href="https://public.tableau.com/app/profile/exafluence2701/viz/Chatprobe/ChatprobeConversationAnalysis",
    target="_blank",
    className='external-link',
    style={
        'color': 'blue',
   ' text-decoration': 'underline',
    'margin': '10px',
    }
),

 ])
]
    ,style={'justify-content': 'center'},className="mt-4"
    )
    ]),
    className="mt-3"
)

 
        ],
            style={'border':'1px solid #dee2e6'}),
            className="mt-3",style={'border':'1px solid #dee2e6','background': '#F5EEC8'}
        )
        tab3_content = dbc.Card(
            dbc.CardBody([
                   dbc.Row(
            [
                dbc.Col(dbc.Card(card_content6, color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'})),
                dbc.Col(
                    dbc.Card(card_content7, color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'})
                ),
                dbc.Col(dbc.Card(card_content8, color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'})),
             ],
            className="",
        ),
         dbc.Row(
            [
                dbc.Col(dbc.Card(card_content9, color=" ", inverse=True,className="cardsbg",
                                #  style={'box-shadow': '1px 1px 4px #dddddd'}
                                 ),style={'max-width': '25%'}),
                dbc.Col(
                    dbc.Card(card_content10, color=" ", inverse=True,className="cardsbg",
                            #  style={'box-shadow': '1px 1px 4px #dddddd'}
                             ),style={'max-width': '25%'}
                ),
                
                 dbc.Collapse(
        dbc.Nav([
            dbc.NavItem(
                html.Div([
    html.Div([
        #  html.Button('Close', id='close-button', n_clicks=0,style={'float':'right'}),
            html.Button('❌', id='close-button', n_clicks=0, style={'float': 'right',
                                                                'font-size': '11px',
                                                                'border': 'none',
                                                                'background': 'none',
                                                                }),
        html.H3('Prompt',style={'text-align':'center','font-size':' 22px','font-weight':'bold'}),
        # html.Table([
        #     html.Tr([html.Th(col),": ", html.Td(selected_row[col])]) for col in df.columns
        # ]),
        # "Document ID:",
            # card_content
            dbc.Card([
            # *[html.P(f"{column}: {selected_row[column]}") for column in df.columns]
            "Prompt"
            ],style={'padding':'0px 0px 23px 13px','background':'rgb(249 249 249)','border':'0px solid black','top': '0.5rem'}
            )
                ], className='card-body')
            ],id='selected-card', className='selected-card',style={'background':'rgb(249 249 249)','height': '15.28rem',
    'overflow-y': 'hidden','box-shadow': 'rgba(0, 0, 0, 0.16) 0px 3px 6px, rgba(0, 0, 0, 0.23) 0px 3px 6px'})
            ),
            # dbc.NavItem(dbc.NavLink("Page 2", href="#")),
            # dbc.NavItem(dbc.NavLink("Page 3", href="#")),
        ], vertical=True, pills=True),
        id="sidebar",
        style={"position": "absolute", "right": 0, "top": 0, "height": "100vh"},
    ),

             ],
            className="mt-4",style={'justify-content': 'center'}
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(
                    dbc.Collapse(
            dbc.Card(dbc.CardBody([
               dash_table.DataTable(
        id='datatable',
        columns=[
                {'name': 'S.no', 'id': 'S.no'},
            {'name': 'Intent', 'id': 'Intent'},
            {'name': 'Subintent', 'id': 'Subintent'},
            {'name': 'Outcome', 'id': 'Outcome'},
            {'name': 'Reason_not_resolved', 'id': 'Reason_not_resolved'}
        ],
        data=[],
        style_header={'backgroundColor': '#89B269', 'fontWeight': 'bold', 'color': '#FFFFFF', 'textAlign': 'center'},
        style_cell={'backgroundColor': '#D3E2C1', 'color': '#000000', 'textAlign': 'center'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#D3E2C1'
            },
            {
                'if': {'row_index': 'even'},
                'backgroundColor': '#A7D397'
            }
        ]
    ),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    ),
             ] ,style={'padding':'0rem','box-shadow': 'rgb(221, 221, 221) 1px 1px 4px'})),
            id="collapse",
            is_open=False,
        ),
                    color=" ", inverse=True,className="cardsbg",style={'box-shadow': '1px 1px 4px #dddddd'}),style={'max-width': '100%'}),
                
             ],
            className="mt-4",style={'justify-content': 'center'}
        )


        ],
            style={'border':'1px solid #dee2e6'}),
            className="mt-3",style={'border':'1px solid #dee2e6','background': '#F5EEC8'}
        )
        tab4_content = html.Div([dbc.Card([
        dbc.CardBody(
            # [                        
            #     html.Div(
            #                 [
            #                     html.H5(
            #                         "Components of the Solution",
            #                         style={"fontSize": "22px", "font-weight": "500"},
            #                     )
            #                 ],style={'box-shadow': '1px 1px 4px #dddddd','text-align': 'center'},
            #                 className="",
            #             ),
            #             dbc.Card(
            #                     dbc.CardBody([
                                   
            #                         html.A(html.Img(src=app.get_asset_url("newarchitecture.png"), 
            #                         className='title-image', style={'width':'100%','height':'100%', 'margin-top':'0px','align':'left', 'margin-Left':'0px', 'paddingLeft':'0px', 'paddingTop':'0px'})),
                                
            #                         html.A("Link to parser platform", href='http://3.239.29.190:9000/parser/homepage', target="_blank")
            #                         ],style={'text-align':'centere'}
            #                     ),
            #                     style={"border": "none"},
            #                 ),

            # ]
        ), 
            
        ],className="mt-3",style={'border':'1px solid #dee2e6'}), 
        ])
        component = dbc.Container( 
                dbc.CardBody(
                    [
                        dbc.Row(
            [  dbc.Col([
            html.Div(
        [
        
        html.Br(),
        dbc.Card(
        dbc.Tabs(
            [
                dbc.Tab( tab3_content, 
                    label="Play Ground",
                    label_style={"color": " white","background-color": "#DADDB1","border-bottom": "3px solid #DADDB1","font-size":"18px"},
                    active_label_style={"color": "black","background-color": "#DE8F5F","border-bottom": "3px solid #3a7099"}, activeTabClassName="fw-bold fst-italic",tab_id="tab-1"
                ,style={'color':'black'}),
                dbc.Tab(tab1_content,
                        label_style={"text-align":"center","color": "white","background-color": "#DADDB1","border-bottom": "3px solid #DADDB1","font-size":"18px"},
                    label="Conversation Analytics",active_label_style={"color": "black ","background-color": "#DE8F5F","border-bottom": "3px solid #3a7099"}, activeTabClassName="fw-bold fst-italic",tab_id="tab-3"
                ,style={'color':'black' }),
                # dbc.Tab(tab4_content,
                #     label="SciFind Info",
                #     label_style={"color": "#404040"},
                #     active_label_style={"color": "#217296","background-color": "#0091ff17","border-bottom": "3px solid #3a7099"}, activeTabClassName="fw-bold fst-italic",tab_id="tab-4"
                # ,style={'color':'black'}),
            ],active_tab="tab-1", className="tabcardbody"
        ),  className="tabcard "
        # ,style={'border': '0px solid rgba(0,0,0,.125)','border-top':'1px solid #dee2e6'}
        
        )
    ]
)
             ])])
                    ],style={'padding': '13px 17px'}
                ),
            fluid=True,
            
        )

        return component
    
    def get_app_layout(self):
        component=html.Div([
      header,
      html.Div([
          data_summary,
        self.get_summary_content(),
      ],id='body')
    ],id='app')
        return component

app_instance=App()
app.layout=app_instance.get_app_layout()

# download sample csv file in forecast
@app.callback(
    Output("download-data1", "data"),
    [Input("Sample-Conversation", "n_clicks")],
    [State("samplefilename1", "value")],
    prevent_initial_call=True,
)
def download_samplefile(_, samplefilepath):
    return dcc.send_file(samplefilepath)

# download sample csv file in forecast
@app.callback(
    Output("download-data2", "data"),
    [Input("Extracted-Output", "n_clicks")],
    [State("samplefilename2", "value")],
    prevent_initial_call=True,
)
def download_samplefile(_, samplefilepath):
    return dcc.send_file(samplefilepath)


# #callback for extract structured data
# # Callback function to be executed on button click
# @app.callback(
#     Output('', 'children'),   
#     Input('ExtractStructuredData', 'n_clicks')  
# )
# def update_output(n_clicks):
     
#     if n_clicks is not None:
         
#         result = " "
#         return result
#     else:
#         return dash.no_update
@app.callback(
    Output("collapse", "is_open"),
    [Input("ExtractStructuredData", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Sample data (You can replace this with dynamic data fetching logic)
def get_dynamic_data():
    # In this example, we'll return a simple DataFrame
    data = {
        'S.no': ['1', '2', '3'],
        'Intent': ['Network-related issue', 'Network-related issue', 'Network-related issue'],
        'Subintent': [['Issue with calls', 'Audio quality (echo)'],
                      ['Internet issue', 'Streaming video'],
                      ['5G issue', 'Check handset 5G ready']],
        'Outcome': ['Resolved by bot', 'Resolved by bot', 'Resolved by bot'],
        'Reason_not_resolved': ['N/A', 'N/A', 'N/A']
    }

    return pd.DataFrame(data)

# Callback function to update the DataTable
@app.callback(
    Output('datatable', 'data'),
    [Input('interval-component', 'n_intervals')]
)
def update_table(n):
    # Fetch dynamic data
    dynamic_data = get_dynamic_data()
    
    # Convert DataFrame to dictionary for DataTable
    data_dict = dynamic_data.to_dict('records')
    
    return data_dict
 
@app.callback(
    Output("sidebar", "is_open"),
    [Input("toggle_sidebar", "n_clicks"), Input("close-button", "n_clicks")],
    [dash.dependencies.State("sidebar", "is_open")],
)
def toggle_sidebar(n_clicks, n_clicks_close, is_open):
    ctx = dash.callback_context
    if not ctx.triggered_id:
        button_id = "No clicks yet"
    else:
        button_id = ctx.triggered_id
    # If the "View Prompt" button is clicked, open the sidebar
    if button_id == "toggle_sidebar":
        return not is_open
    # If the "❌" button is clicked, close the sidebar
    elif button_id == "close-button":
        return False
    # Default case, return the current state
    return is_open
 


if __name__ == '__main__':
  app.run_server('0.0.0.0',port=3016,debug=False)