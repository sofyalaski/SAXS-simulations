#!/bin/bash

#run spheres in sizes [81, 125,243,343,441,625,729,875,945] in dtype 32 and 64, each 5 times to compare memory, time and error

shapes=("cylinder" "sphere" "hardsphere")
sizeValues=(81 125 )
dTypes=(32 64 243 343 441 625 729 875 945)
END=5
for shape in ${shapes[@]}; do
    for s in ${sizeValues[@]}; do
        for d in ${dTypes[@]}; do 
            for ((i=1;i<=END;i++)); do
                echo "run  $shape in size $s and data type $d $i -th time$"
                python3 benchmarking/test_fft.py $shape $s $d
            done
        done
    done
done    


