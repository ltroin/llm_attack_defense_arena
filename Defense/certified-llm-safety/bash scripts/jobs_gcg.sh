
#!/bin/bash

# Loop through the range of num_adv values and run gcg.py for each value
for num_adv in 14 16 18 # 20 0 2 4 6 8 10 12
do
    python gcg.py --num_adv $num_adv
done
