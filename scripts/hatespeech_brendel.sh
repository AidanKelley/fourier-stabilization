#BSUB -o out/hatespeech_brendel.%J
#BSUB -e out/hatespeech_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d hatespeech -i models/hatespeech.h5:custom_sigmoid \
  --all -o attack_data/hatespeech_brendel.json -m pdf
