#BSUB -o out/pdfrate_at5_stable_custom_jsma.%J
#BSUB -e out/pdfrate_at5_stable_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J pdfrate_at5_stable_custom_jsma

cd ~/codnn
source env/bin/activate

python attack.py custom_jsma \
  -d pdfrate -i models_at/pdfrate_at5_stable.h5:custom_sigmoid \
  --all -o attack_data_at/pdfrate_at5_stable_custom_jsma.json -m pdf
