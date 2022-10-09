for d in graphs/*/ ; do
    python3 graphs/$(basename "$d")/$(basename "$d").py
    for f in "$d"*.onnx ; do
        python3 modelParserONNX.py -f "$f"
        make graphs
    done
    python3 graphs/timesPlot.py
    rm graphs/timesF.txt
done
