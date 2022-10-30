python3 graphs/conv/conv.py
for inpSize in 0 1 2 3 4 ; do
    for kernelDim in 0 1 2 3 ; do
        python3 graphs/conv/convf90.py "$inpSize" "$kernelDim"
        python3 modelParserONNX.py -f "$(ls -Art1 graphs/conv/*.onnx | tail -n 1)"
        make graphs
    done
done
python3 graphs/timesPlot.py "conv"
rm graphs/timesF.txt
