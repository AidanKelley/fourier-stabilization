#BSUB -o out/fraud_stable_0.98_custom_jsma.%J
#BSUB -e out/fraud_stable_0.98_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J fraud_stable_0.98_custom_jsma

cd ~/codnn
source env/bin/activate

python attack.py custom_jsma \
  -d fraud -i models/fraud_stable_0.98.h5:custom_sigmoid \
  --all -o attack_data2/fraud_stable_0.98_custom_jsma.json -m pdf
