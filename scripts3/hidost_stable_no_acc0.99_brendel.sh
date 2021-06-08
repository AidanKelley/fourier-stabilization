#BSUB -o out/hidost_stable_no_acc0.99_brendel.%J
#BSUB -e out/hidost_stable_no_acc0.99_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_stable_no_acc0.99_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d hidost_scaled -i models3/hidost_stable_no_acc0.99.h5:custom_sigmoid \
  --all -o attack_data3/hidost_stable_no_acc0.99_brendel.json -m pdf
