#BSUB -o out/hatespeech_stable_0.90_custom_jsma.%J
#BSUB -e out/hatespeech_stable_0.90_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hatespeech_stable_0.90_custom_jsma

cd ~/codnn
source env/bin/activate

python attack.py custom_jsma \
  -d hatespeech -i models/hatespeech_stable_0.90.h5:custom_sigmoid \
  --all -o attack_data2/hatespeech_stable_0.90_custom_jsma.json -m pdf
