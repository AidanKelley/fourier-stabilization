#BSUB -o out/hidost_0.975_custom_jsma.%J
#BSUB -e out/hidost_0.975_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_0.975_custom_jsma

cd ~/codnn
source env/bin/activate

python attack.py custom_jsma \
  -d hidost_scaled -i models_relu/hidost_0.975.h5:relu \
  --all -o attack_data_relu/hidost_0.975_custom_jsma.json -m pdf
