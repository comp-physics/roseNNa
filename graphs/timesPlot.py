import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def plot(times):
    fig = px.scatter(times, 
        x="Layers", 
        y="Time", 
        color="Neuron/Layer", 
        title="Python MLP Times")
    # fig.write_image("mlp.png")
    fig.show()

with open("graphs/timesF.txt") as f, open("graphs/mlp/times.txt") as f2:
    fortran = f.readlines()
    py = f2.readlines()
    outputF = []
    for x in fortran:
        outputF.append(float(x.strip()))
    outputP = list(map(float,py[0].strip().split(" ")))
    div = np.divide(np.array(outputF),np.array(outputP))
    dic = []
    d = 0
    for layer in [1,5,10,25,50]:
        for neurons in [10,25,50,100]:
            dic.append({"Layers":layer,"Time":div[d],"Neuron/Layer":str(neurons)})
            d+=1
    pl = pd.DataFrame(dic)
    pl.to_csv("mlp_times.csv")
    plot(pl)