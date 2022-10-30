python3 graphs/maxpool/maxpool.py
for inpSize in 0 1 2 3 4 ; do
    for kernelDim in 0 1 2 3 ; do
        python3 graphs/maxpool/maxpoolf90.py "$inpSize" "$kernelDim"
        python3 modelParserONNX.py -f "$(ls -Art1 graphs/maxpool/*.onnx | tail -n 1)"
        make graphs
    done
done
python3 graphs/timesPlot.py "maxpool"
rm graphs/timesF.txt
