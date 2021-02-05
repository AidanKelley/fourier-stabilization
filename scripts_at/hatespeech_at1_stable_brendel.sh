#BSUB -o out/hatespeech_at1_stable_brendel.%J
#BSUB -e out/hatespeech_at1_stable_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hatespeech_at1_stable_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d hatespeech -i models_at/hatespeech_at1_stable.h5:custom_sigmoid \
  --all -o attack_data_at/hatespeech_at1_stable_brendel.json -m pdf
