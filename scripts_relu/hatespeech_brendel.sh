#BSUB -o out/hatespeech_brendel.%J
#BSUB -e out/hatespeech_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hatespeech_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d hatespeech -i models_relu/hatespeech.h5:relu \
  --all -o attack_data_relu/hatespeech_brendel.json -m pdf
