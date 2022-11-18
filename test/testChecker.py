import sys
import os
file = sys.argv[1]
with open("test.txt") as f, open("../goldenFiles/"+file+"/"+file+".txt") as f2:
    try:
        fortran = f.readlines()
        py = f2.readlines()
        shapeF = list(map(int,fortran[0].strip().split()))
        outputF = list(map(float,fortran[1].strip().split()))
        shapeP = list(map(int,py[0].strip().split()))
        outputP = list(map(float,py[1].strip().split()))
        outShape = shapeP == shapeF
        outRes = True
        for p, f in zip(outputP,outputF):
            if abs(p-f) > 10**-5:
                outRes = False
                break
    except Exception as e:
        print(str(e))
    finally:
        outCompRun = True
        outputFailPath = "outputCase.txt"
        if os.path.exists(outputFailPath) and os.stat(outputFailPath).st_size != 0:
            outCompRun = False
        if outCompRun:
            if outRes and outShape:
                print("Outputs match! Pass!")
            else: #shapes do not match
                print("Fail!!")
                if not outShape:
                    print("Output shapes do not match!!")
                    print(f"Correct shape is {shapeP}. But, F90 outputted {shapeF}")
                if not outRes:
                    print("Incorrect outputs.")
                sys.exit(1)
        else:
            print("Error occurred while executing! Failed! Here is the output: ")
            failed = open("outputCase.txt",'r')
            fail = failed.read()
            print(fail)
            sys.exit(1)
