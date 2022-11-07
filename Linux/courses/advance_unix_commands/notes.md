
# Fetch data

Fetch column 1, 2 and 3
´´´bash
awk '{print $1 $2 $3}' DataFileSpace.tx
´´´

Fetch columns 1, 2 and 4 in different order and seperate the output with " , ".
´´´bash
awk '{print $2 " , "  $1 " , " $4}' DataFileSpace.tx
´´´
