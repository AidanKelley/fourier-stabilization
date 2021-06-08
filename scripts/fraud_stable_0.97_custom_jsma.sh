#BSUB -o out/fraud_stable_0.97_custom_jsma.%J
#BSUB -e out/fraud_stable_0.97_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d fraud -i models/fraud_stable_0.97.h5:custom_sigmoid \
  --all -o attack_data/fraud_stable_0.97_custom_jsma.json -m pdf
