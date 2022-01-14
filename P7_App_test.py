# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
from operator import itemgetter
from sklearn.neighbors import NearestNeighbors

# Lecture des données non étiquettés, i.e à prédire
app_test = pd.read_csv('app_test_1000.csv', sep=',', index_col=0, encoding='utf8')

# Lecture des données de validation
conf_mx = pickle.load(open('conf_mx_ind_bk.md', 'rb'))

# Lecture des données non étiquettés, brutes i.e non traités
app_test_no_transformation = pd.read_csv('app_test_no_transformation_1000.csv',sep=',',index_col=0,encoding='utf8')

# Lecture du modèle
clf_pipe = pickle.load(open('banking_model.md', 'rb'))

x_test_transformed = pd.DataFrame(clf_pipe[0].transform(app_test.drop(columns=["TARGET"])),
                                  columns = app_test.drop(columns=["TARGET"]).columns,
                                  index = app_test.index)
# Calcul des 20 plus proches voisins
nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(x_test_transformed)

# Interprétabilité du modèle
lime1 = LimeTabularExplainer(x_test_transformed,
                             feature_names = x_test_transformed.columns,
                             class_names = ["Solvable", "Non Solvable"],
                             discretize_continuous = False)

def feature_importances(n_top_features = 20):
    
    indices, values = [], []

    for ind, val in sorted(zip(clf_pipe[1].feature_importances_,
                               x_test_transformed.columns), reverse=True)[0:  n_top_features] :
        indices.append(ind)
        values.append(val)
    data = pd.DataFrame(values, columns=["values"], index=indices)
    del indices, values
    
    return {
        'data': [go.Bar(
                    x=data.index,
                    y=data["values"],
                    orientation='h',
        )],
        
        'layout': go.Layout(
                            margin={'l': 300, 'b': 50, 't': 30, 'r': 30},
                            height=700,
                            width=1200,
                           )
    }

def plot_mat_conf(conf_mx):
    
    labels = ["Solvable", "Non Solvable"]
    
    annotations = go.Annotations()
    for n in range(conf_mx.shape[0]):
        for m in range(conf_mx.shape[1]):
            annotations.append(go.Annotation(text=str(conf_mx[n][m]), x=labels[m], y=labels[n],
                                             showarrow=False))

    trace = go.Heatmap(x=labels,
                       y=labels,
                       z=conf_mx,
                       colorscale='Viridis',
                       showscale=False)

    fig = go.Figure(data=go.Data([trace]))
    fig['layout'].update(
        annotations=annotations,
        xaxis= dict(title='Classes prédites'), 
        yaxis=dict(title='Classes réelles', dtick=1),
        margin={'b': 30, 'r': 20, 't': 10},
        width=700,
        height=500,
        autosize=False
    )
    
    return fig # Retourne la figure crée

num_columns = app_test_no_transformation.select_dtypes(include=["float64"]).columns

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    
    dcc.Tabs([
        # Premier onglet: Solvability Client
        dcc.Tab(label='Solvability Client', children=[
            # Permet de séléctionner dans une liste déroulante le numéro du client
            html.Div([
                html.H3("Id Client"),
                dcc.Dropdown(
                id='id-client',
                options=[{'label': i, 'value': i} for i in x_test_transformed.index],
                value=x_test_transformed.index[0]
                ),
            ]),
            html.Div([
                # Affiche la probabilité de solvabilité d'un client
                # sous forme de pie plot
                html.Div([
                    html.H3("Probability of Solvability Client"),
                    dcc.Graph(id='proba',
                              figure={},
                              style={"height": 500,
                                     "width": 500}
                             ),
                ], className='six columns'),
                # Affiche pour l'id client séléctionné
                # l'importance des features qui ont eu le plus d'impacte
                # sur la solvabilité d'un client ou non
                html.Div([
                    html.H3("Feature Importances"), 
                    dcc.Graph(id='graph',
                              figure={},
                              style={"height":500,
                                     "width":800}
                             ),       
                ], className='six columns'),        
            ], className="row"),
            # Affiche un tableau contenant les informations relatives
            # au client séléctionné ainsi que les clients sililaires
            html.Div([
                html.H3("Similary Clients"),
                dash_table.DataTable(
                    id='table',
                    columns=[
                       {"name": i, "id": i} for i in app_test_no_transformation.reset_index().columns
                    ],
                    filter_action='custom',
                    filter_query='',
                    fixed_rows={'headers': True, 'data': 0 },
                    style_cell={'width': '200px'},
                    style_table={'minWidth': '80%'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                }, 
                    virtualization=True,
                ), 
            ], className='row'),
                
        ]),
        # Deuxieme Onglet : Model Performance
        dcc.Tab(label="Model Performance", children=[
            html.Div([
                # Affiche la matrice de confusion obtenue
                # sur les données test
                html.Div([
                    html.H3("Confusion Matrix"),
                    dcc.Graph(id='cf_mat',
                              figure= plot_mat_conf(conf_mx),
                             ),
                ], className='six columns'),
                # Affiche les feature importances globables, i.e celles
                # qui ont le plus d'importance sur la prédiction du modèle
                html.Div([
                    html.H3("Feature Importances"), 
                    dcc.Graph(id='graph_feature',
                              figure=feature_importances()),   
                ], className="six columns"),
            ]),
        ]),
    
        # Troisième onglet
        dcc.Tab(label='Data exploration', children=[
           html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='xaxis-column',
                        options=[{'label': i, 'value': i} for i in num_columns],
                        value='AMT_CREDIT'
                    ),
                    dcc.RadioItems(
                        id='xaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ],
                style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Dropdown(
                        id='yaxis-column',
                        options=[{'label': i, 'value': i} for i in num_columns],
                        value='AMT_ANNUITY'
                    ),
                    dcc.RadioItems(
                        id='yaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),

            dcc.Graph(id='indicator-graphic'),

        ]),

    ]),
])

    
# Création d'un système de filtre
operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]

def split_filter_part(filter_part):
    # Permet d'avoir un outil de filtrage des données
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


# Met à jour le tableau de données
# Le tableau correspond aux clients similaires de l'id client choisie
@app.callback(
    Output('table', 'data'),
    [Input('table', "filter_query"),
     Input('id-client', "value")])
def update_table(filter, id_client):
    
    # Déterminer les individus les plus proches du client dont l'id est séléctionné
    indices_similary_clients = nbrs.kneighbors(np.array(x_test_transformed.loc[id_client]).reshape(1, -1))[1].flatten()
     
    filtering_expressions = filter.split(' && ')
    dff = app_test_no_transformation.iloc[indices_similary_clients].reset_index()
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]
    
    return dff.to_dict('records')


# Met à jour le pieplot de la solvabilité du client dont l'id est choisie
@app.callback(
    Output('proba', 'figure'),
    [Input('id-client', 'value')])
def proba_pie(id_client):
    
    values = clf_pipe[1].predict_proba(np.array(x_test_transformed.loc[id_client]).reshape(1, -1)).flatten()
        
    # Retourne le pie plot mis à jour pour l'id client
    return {
        'data': [go.Pie(labels=['Solvable', "Non Solvable"],
                        values=values,
                        marker_colors=["#2ecc71", "#e74c3c"],
                        hole=.5
                       )],
        'layout': go.Layout(margin=dict(b=100)
                           )
    }
    del values
    
    
    
@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('xaxis-type', 'value'),
     Input('yaxis-type', 'value')])
def update_graph_2(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type):
       
    traces = []
    solvable_labels = ["Solvable", "Non Solvable"]
    for i, target in enumerate(app_test_no_transformation.TARGET.unique()):
        filtered_df = app_test_no_transformation[app_test_no_transformation['TARGET'] == target].reset_index()
        traces.append(dict(
            x=filtered_df[xaxis_column_name],
            y=filtered_df[yaxis_column_name],
            text=filtered_df['SK_ID_CURR'],
            mode='markers',
            opacity=0.7,
            marker={
                'color':list(filtered_df["TARGET"].map({0.0: '#e74c3c', 1.0: "#2ecc71"}).values),
                'size': 5,
                'line': {'width': 0.15, 'color': 'white'}
            },
            name=solvable_labels[i]
        ))   
        
    return {
        'data': traces,
        'layout': dict(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }
      
        
# Met à jour le graphique de l'importance des features pour 
# le client dont l'id est séléctionné
@app.callback(
    Output('graph', 'figure'),
    [Input('id-client', 'value'),
    ])
def update_graphic(id_client) :
     
    exp = lime1.explain_instance(x_test_transformed.loc[id_client],
                                 clf_pipe[1].predict_proba,
                                 num_samples=100)
    
    indices, values = [], []
    

    for ind, val in sorted(exp.as_list(), key=itemgetter(1)):
        indices.append(ind)
        values.append(val)
    data = pd.DataFrame(values, columns=["values"], index=indices)
    data["positive"] = data["values"]>0
    del indices, values
    
    # Retourne le barplot correspondant aux 'feature importances'
    # du client dont l'id est séléctionné sur le dashboard
    return {
        
        'data': [go.Bar(
                    x=data["values"],
                    y=data.index,
                    orientation='h',
                    marker_color=list(data.positive.map({True: '#2ecc71', False: '#e74c3c'}).values)
        )],
        
        'layout': go.Layout(
                            margin=dict(l=300, r=0, t=30, b=100)
                           )
    } 

    
if __name__ == '__main__':
    app.run_server(debug = False)
