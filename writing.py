import os
from pathlib import Path
arr1 = [2,234,4,5,6,6]
with open("tes.fpp",'w') as f:
    f.write(f"""#:set arr1 = {arr1}""")
    f.write("\n")
    f.write("a")

