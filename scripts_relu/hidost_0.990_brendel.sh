#BSUB -o out/hidost_0.990_brendel.%J
#BSUB -e out/hidost_0.990_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_0.990_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d hidost_scaled -i models_relu/hidost_0.990.h5:relu \
  --all -o attack_data_relu/hidost_0.990_brendel.json -m pdf
