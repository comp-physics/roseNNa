import sys
import os
file = sys.argv[1]
with open("goldenFiles/test.txt") as f, open("goldenFiles/"+file+"/"+file+".txt") as f2:
    fortran = f.readlines()
    py = f2.readlines()
    shapeF = list(map(int,fortran[0].strip().split()))
    outputF = list(map(float,fortran[1].strip().split()))
    shapeP = list(map(int,py[0].strip().split()))
    outputP = list(map(float,py[1].strip().split()))
    out = shapeP == shapeF
    for p, f in zip(outputP,outputF):
        if abs(p-f) > 10**-6:
            out = False
            break
    outCompRun = True
    outputFailPath = "/Users/ajaybati/Documents/researchcompphys/outputCase.txt"
    if os.path.exists(outputFailPath) and os.stat(outputFailPath).st_size != 0:
        outCompRun = False
    if outCompRun:
        if out:
            print("Outputs match! Pass!")
        else:
            print("Outputs do not match! Fail!")
            sys.exit(1)
    else:
        print("Error occurred while executing! Failed! Here is the output: ")
        failed = open("/Users/ajaybati/Documents/researchcompphys/outputCase.txt",'r')
        fail = failed.read()
        print(fail)
        sys.exit(1)