#BSUB -o out/hatespeech_at1_custom_jsma.%J
#BSUB -e out/hatespeech_at1_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hatespeech_at1_custom_jsma

cd ~/codnn
source env/bin/activate

python attack.py custom_jsma \
  -d hatespeech -i models_at/hatespeech_at1.h5:custom_sigmoid \
  --all -o attack_data_at/hatespeech_at1_custom_jsma.json -m pdf
