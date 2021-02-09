#BSUB -o out/hatespeech_0.87_custom_jsma.%J
#BSUB -e out/hatespeech_0.87_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hatespeech_0.87_custom_jsma

cd ~/codnn
source env/bin/activate

python attack.py custom_jsma \
  -d hatespeech -i models_relu/hatespeech_0.87.h5:relu \
  --all -o attack_data_relu/hatespeech_0.87_custom_jsma.json -m pdf
