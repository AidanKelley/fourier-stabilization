#BSUB -o out/hidost_stable_0.993_brendel.%J
#BSUB -e out/hidost_stable_0.993_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_stable_0.993_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d hidost_scaled -i models/hidost_stable_0.993.h5:custom_sigmoid \
  --all -o attack_data2/hidost_stable_0.993_brendel.json -m pdf
