npass=0
nfail=0
testnum=1
skip="__pycache__"
make compile
make unit || { echo "FATAL: Fortran unit tests failed"; exit 1; }
python3 test_parser.py || { echo "FATAL: parser tests failed"; exit 1; }
for d in ../goldenFiles/*/ ; do
    name=$(basename "$d")
    if [[ "$name" != "$skip" ]] && [[ "$name" != "vgg16" ]] && [[ "$name" != "gemm_huge" ]] && [[ "$name" != "turbulentShear" ]]; then
        echo "---------------- TEST #$testnum $name -------------------"
        if make test case="$name"; then
            output=$(python3 -Wi testChecker.py "$name")
            if [[ $? -eq 0 ]]; then
                ((++npass))
            else
                ((++nfail))
            fi
            echo "$output"
        else
            # test.txt is stale; skip the checker
            ((++nfail))
            echo "Fail!! make test failed for $name (parser rejection or build failure); outputs not compared"
        fi
        echo -e "---------------- TEST #$testnum $name -------------------\n"
        ((++testnum))
    fi
done
echo "$npass out of $(($npass + $nfail)) test cases have passed!"
exit $nfail
