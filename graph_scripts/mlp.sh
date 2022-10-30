python3 graphs/mlp/mlp.py
for f in graphs/mlp/*.onnx ; do
    echo "$f"
    python3 modelParserONNX.py -f "$f"
    make graphs
done
python3 graphs/timesPlot.py "mlp"
rm graphs/timesF.txt
