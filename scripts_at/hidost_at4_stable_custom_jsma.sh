#BSUB -o out/hidost_at4_stable_custom_jsma.%J
#BSUB -e out/hidost_at4_stable_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_at4_stable_custom_jsma

cd ~/codnn
source env/bin/activate

python attack.py custom_jsma \
  -d hidost_scaled -i models_at/hidost_at4_stable.h5:custom_sigmoid \
  --all -o attack_data_at/hidost_at4_stable_custom_jsma.json -m pdf
