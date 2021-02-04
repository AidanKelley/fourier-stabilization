#BSUB -o out/hatespeech_stable_no_acc0.88_brendel.%J
#BSUB -e out/hatespeech_stable_no_acc0.88_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hatespeech_stable_no_acc0.88_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d hatespeech -i models3/hatespeech_stable_no_acc0.88.h5:custom_sigmoid \
  --all -o attack_data3/hatespeech_stable_no_acc0.88_brendel.json -m pdf
