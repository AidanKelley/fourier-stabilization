#BSUB -o out/hidost_stable_0.996_custom_jsma.%J
#BSUB -e out/hidost_stable_0.996_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_stable_0.996_custom_jsma

cd ~/codnn
source env/bin/activate

python attack.py custom_jsma \
  -d hidost_scaled -i models/hidost_stable_0.996.h5:custom_sigmoid \
  --all -o attack_data2/hidost_stable_0.996_custom_jsma.json -m pdf
