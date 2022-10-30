import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import sys

def plot(times):
    cols = times.columns
    fig = px.scatter(times,
        x=times[cols[1]],
        y=times[cols[2]],
        color=times[cols[3]].astype(str),
        title="Time Ratios (F90/Python)")
    # fig.write_image("mlp.png")
    fig.show()

a = pd.read_csv("conv/conv_times.csv")
print(a)
plot(a)