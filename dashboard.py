import os
import pickle
import uuid

import numpy as np
import matplotlib.pyplot as plt

import backend
from dash.dependencies import Input, Output
import pandas
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import json
import dash_table
import plotly.graph_objs as go
import dash_reusable_components as drc
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance

app = dash.Dash(__name__)
server = app.server

N = backend.NetworkWrapper()

def addVectSelectors():
    return dash_table.DataTable(
        id='vec_dims',
        columns=[{"name": i, "id": i} for i in range(8)],
        data={i : [0 for j in range(64)] for i in range(8)},
    )

df = pandas.DataFrame({i : [0 for j in range(64)] for i in range(8)}, columns=range(8))
print(df)
print(df.to_dict('rows'))
print(df.columns)


def serve_layout():
    # Generates a session ID
    session_id = str(uuid.uuid4())

    # App Layout
    return html.Div([
        # Session ID
        html.Div(session_id, id='session-id', style={'display': 'none'}),
        # Banner display
        html.Div([
            html.H2(
                'stylegan',
                id='title'
            ),
            html.Img(
                src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"
            )
        ],
            className="banner"
        ),

        # Body
        html.Div(
            className="container", children=[
            html.Div(className='row', children=[
                html.Div(className='five columns',
                    children=[
                        dash_table.DataTable(
                            id='vec_dims',
                            editable=True,
                            data=[{j : 0 for j in range(8)} for i in range(64)],
                            columns=[{'id': c, 'name': c} for c in range(8)],
                            n_fixed_rows=1,
                            style_cell={'width': '20px'},
                        )
                        #dash_table.DataTable(
                        #    id='vec_dims',
                        #    columns=[{"name": i, "id": i} for i in range(8)],
                        #    data={i : [0 for j in range(10)] for i in range(8)},
                        #)
                    ]),
                html.Div(
                    className='seven columns',
                    style={'float': 'right'},
                    children=[
                        html.Div(id='div-interactive-image', children=[
                            drc.InteractiveImagePIL(
                                image_id='interactive-image',
                                image=N.genRandomImage(),
                                )
                            ]),
                    ]),
            ]),
        ]),
    ])





app.layout = serve_layout

@app.callback(
    Output('div-interactive-image', 'children'),
    [Input('vec_dims', 'data'),
     Input('vec_dims', 'columns')])
def display_output(rows, columns):

    vec = []
    for r in rows:
        vec += [v for k, v in r.items()]
    print(np.array(vec).reshape(1,-1))
    print(np.array(vec).reshape(1,-1).shape)
    return [drc.InteractiveImagePIL(
        image_id='interactive-image',
        image=N.genImage(np.array(vec).reshape(1,-1)),
        )]

def main():
    print("Generating")
    app.run_server(debug=True, port=9012,host='0.0.0.0')

if __name__ == '__main__':
    main()
