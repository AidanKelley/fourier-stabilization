#BSUB -o out/fraud_stable_no_acc0.97_brendel.%J
#BSUB -e out/fraud_stable_no_acc0.97_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J fraud_stable_no_acc0.97_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d fraud -i models3/fraud_stable_no_acc0.97.h5:custom_sigmoid \
  --all -o attack_data3/fraud_stable_no_acc0.97_brendel.json -m pdf
