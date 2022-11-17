import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def plot(times):
    cols = times.columns
    colors = times[cols[-2]].unique()
    allX = []
    allY = []
    names = []
    for x in colors:
        curr_color = times[times[cols[-2]]==x]
        allX.append(list(map(int,curr_color[cols[1]].tolist())))
        allY.append(curr_color[cols[2]].tolist())
        names.append(str(x))
            
    for a,b, name in zip(allX, allY, names):
        plt.plot(a, b, label = name)
    
    plt.legend(bbox_to_anchor=(1.0, 1.1),loc="upper right", ncol=len(colors)).get_frame().set_linewidth(0.0)

    plt.show()

a = pd.read_csv("lstm/lstm_times.csv")
plot(a)