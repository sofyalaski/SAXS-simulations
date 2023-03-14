#!/bin/bash

#run spheres in sizes [81, 125,243,343,441,625,729,875,945] in dtype 32 a, each 5 times to compare Chi squared metrics based on diferent uncertainty criteria

shapes=("cylinder" "sphere" "hardsphere")
sizeValues=(81 125 243 343 441 625 729 875 945)
END=5
for shape in ${shapes[@]}; do
    for s in ${sizeValues[@]}; do
        for ((i=1;i<=END;i++)); do
            echo "run  $shape in size $s $i -th time$"
            python3 benchmarking/test_chi2.py $shape $s
        done
    done
done    


