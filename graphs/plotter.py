# import plotly.graph_objects as go
# import plotly.express as px
# import numpy as np
# import pandas as pd
# import sys

# def plot(times):
#     cols = times.columns
#     colors = times[cols[-2]].unique()
#     allX = []
#     allY = []
#     names = []
#     texts = []
#     for x in colors:
#         curr_color = times[times[cols[-2]]==x]
#         allX.append(list(map(int,curr_color[cols[1]].tolist())))
#         allY.append(curr_color[cols[2]].tolist())
#         names.append(str(x))
#         if not curr_color[cols[-1]].dropna().empty:
#             texts.append(curr_color[cols[-1]].tolist())
#         else:
#             texts.append(list(map(int,curr_color[cols[1]].tolist())))

#     fig = go.Figure()
#     for a,b, name, text in zip(allX, allY, names,texts):
#         fig.add_trace(go.Scatter(x=a, y=b,
#                         mode='lines+markers+text',
#                         text = text,
#                         textposition="bottom right",
#                         name=name))
    
#     # fig.add_trace(go.Scatter(x=[int(4)], y=[1.231689],
#     #                     mode='markers+text',
#     #                     text=["Freund et al., 4"],textposition="top right", name = "128", textfont=dict(size=18), showlegend=True))
    
#     # fig.add_trace(go.Scatter(x=[int(6)], y=[0.092527574],
#     #                     mode='markers+text',
#     #                     text=["Zhang et al., 6"],textposition="top right", name = "16", textfont=dict(size=18), showlegend=True))
    

#     fig.update_layout(legend_title={'text': "Neurons/Layer", 'font':dict(size=25)}, legend = {'font':dict(size=15)})
#     fig.update_layout(
#         xaxis_title={'text':"Number of Layers", 'font':dict(size=25,color='#000000')},
#         xaxis = dict(tickfont=dict(size=25, color='black')),
#         yaxis = dict(tickfont=dict(size=25, color='black')),
#         yaxis_title={'text':"Times (F90/Python)", 'font':dict(size=25,color='#000000')})
#     fig.update_layout(autosize=True)
#     fig.update_traces(marker=dict(size=12))
    
#     fig.show()

# a = pd.read_csv("mlp/mlp_times.csv")
# # print(a)
# plot(a)


import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import sys

def plot(times):
    cols = times.columns
    colors = times[cols[-2]].unique()
    allX = []
    allY = []
    names = []
    texts = []
    for x in colors:
        curr_color = times[times[cols[-2]]==x]
        allX.append(list(map(int,curr_color[cols[1]].tolist())))
        allY.append(curr_color[cols[2]].tolist())
        names.append(str(x))
        if curr_color[cols[-1]].dropna().empty:
            texts.append(list(map(int,curr_color[cols[1]].tolist())))
        else:
            texts.append([""])
            

    fig = go.Figure()
    for a,b, name, text in zip(allX, allY, names, texts):
        fig.add_trace(go.Scatter(x=a, y=b,
                        mode='lines',
                        name=name,line=dict(width=3)))
    
    # fig.add_trace(go.Scatter(mode='markers+text',x=[5], y=[0.01069197853], #5 sequence length, 90 hid dim
    #                      text=["Srinivasan et al. (2019)"],textposition="middle right", textfont=dict(size=18,color='#000000'),
    #                      showlegend=False))
    
    # fig.add_trace(go.Scatter(x=[25], y=[0.10693213340314779], #25 seq length, 20 hid dim
    #                     text=["Li et al. (2020)"],textposition="middle right", textfont=dict(size=18,color='#000000'),
    #                     mode='markers+text', showlegend=False))
    fig.add_trace(go.Scatter(mode='markers+text',x=[1], y=[0.057490920835847774], #4,5,3: 1 middle layer with 5 neurons
                             text=["Zhou et al. (2019)"],textposition="bottom right", textfont=dict(size=18,color='#000000'),
                             showlegend=False))
    
    fig.add_trace(go.Scatter(x=[6], y=[0.092527574], #6 layers, 16 neurons/layer
                        text=["Zhang et al. (2020)"],textposition="top right", textfont=dict(size=18,color='#000000'),
                        mode='markers+text', showlegend=False))

    fig.update_layout(legend = {'font':dict(size=20)},
                      autosize=False,
                      width=710,
                      height=530)

    
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_layout(
        xaxis = dict(tickfont=dict(size=25, color='black'),zeroline=False),
        yaxis = dict(tickfont=dict(size=25, color='black'),zeroline=False),
        xaxis_title={'text':"Number of Layers", 'font':dict(size=25,color='#000000')},
        yaxis_title={'text':"Times (roseNNa/PyTorch)", 'font':dict(size=25,color='#000000')},
        plot_bgcolor='white')
    fig.update_traces(marker=dict(size=20))
    fig.update_layout(font_family="Computer Modern",
    title_font_family="Computer Modern", font_color = "black",
    legend_title_font_family="Computer Modern",legend_font_family="Computer Modern",legend_title={'text': "Neurons/Layer", 'font':dict(size=25,color='#000000')},
                      legend=dict(orientation = 'h', xanchor = "center", x = 0.56, y= 1.12,bgcolor = 'rgba(0,0,0,0)'))
    
    fig.write_image("mlp.pdf")

    fig.show()

a = pd.read_csv("mlp/mlp_times.csv")
plot(a)
