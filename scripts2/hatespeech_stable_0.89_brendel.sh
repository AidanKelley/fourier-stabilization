#BSUB -o out/hatespeech_stable_0.89_brendel.%J
#BSUB -e out/hatespeech_stable_0.89_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hatespeech_stable_0.89_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d hatespeech -i models/hatespeech_stable_0.89.h5:custom_sigmoid \
  --all -o attack_data2/hatespeech_stable_0.89_brendel.json -m pdf
