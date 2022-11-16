import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import sys
#mlp, conv, maxpool, lstm
print(sys.argv)
dir_check = sys.argv[1]

def plot(times):
    cols = times.columns
    fig = px.scatter(times,
        x=cols[0],
        y=cols[1],
        color=cols[2],
        title="Time Ratios (F90/Python)")
    # fig.write_image("mlp.png")
    fig.show()
dir = "graphs/"+dir_check+"/times.txt"
with open("graphs/timesF.txt") as f, open(dir) as f2:
    fortran = f.readlines()
    py = f2.readlines()
    outputF = []
    for x in fortran:
        outputF.append(float(x.strip()))
    outputP = list(map(float,py[0].strip().split(" ")))
    div = np.divide(np.array(outputF),np.array(outputP))
    dic = []
    d = 0
    if dir_check == "mlp":
        for layer in [1,5,10,25,50]:
            for neurons in [10,25,50,100]:
                dic.append({"Layers":layer,"Time":div[d],"Neuron/Layer":str(neurons)})
                d+=1
        pl = pd.DataFrame(dic)
        pl.to_csv("graphs/mlp/mlp_times.csv")
    elif dir_check == "conv" or dir_check == "maxpool":
        for inpSize in [30,50,100,500,1000]:
            for neurons in [3,9,15,25]:
                dic.append({"Input Size":str(inpSize), "Time":div[d],"Kernel Size/Inp Size":str(neurons)})
                d+=1
        pl = pd.DataFrame(dic)
        pl.to_csv(f"graphs/{dir_check}/{dir_check}_times.csv")
    elif dir_check == "lstm":
        for inpSize in [2,5,10,50,100]:
            for neurons in [3,25,50,75]:
                dic.append({"Sequence Length":str(inpSize), "Time":div[d],"Hidden Dim/Sequence Length":str(neurons)})
                d+=1
        pl = pd.DataFrame(dic)
        pl.to_csv(f"graphs/{dir_check}/{dir_check}_times.csv")
        plot(pl)
