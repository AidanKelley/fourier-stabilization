#BSUB -o out/hidost_similarity_job.%J
#BSUB -e out/hidost_similarity_job.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_similarity_job

cd ~/codnn/pdflearning
pipenv run python weight_similarity.py hidost -a custom_sigmoid -o similarity/hidost_train_64_1248x100k.json -n 64 -N 100000 -N 200000 -N 400000 -N 800000

