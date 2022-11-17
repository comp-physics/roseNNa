python3 graphs/lstm/lstm.py
for inpSize in 0 1 2 3 4 ; do
    for kernelDim in 0 1 2 3 ; do
        python3 graphs/lstm/lstmf90.py "$inpSize" "$kernelDim"
        echo "$(ls -Art1 graphs/lstm/*.onnx | tail -n 2 | head -1)"
        python3 modelParserONNX.py -f "$(ls -Art1 graphs/lstm/*.onnx | tail -n 2 | head -1)" -w "$(ls -Art1 graphs/lstm/*.onnx | tail -n 2)"
        make graphs
    done
done
python3 graphs/timesPlot.py "lstm"
rm graphs/timesF.txt
