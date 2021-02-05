#BSUB -o out/hidost_at1_brendel.%J
#BSUB -e out/hidost_at1_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_at1_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d hidost_scaled -i models_at/hidost_at1.h5:custom_sigmoid \
  --all -o attack_data_at/hidost_at1_brendel.json -m pdf
