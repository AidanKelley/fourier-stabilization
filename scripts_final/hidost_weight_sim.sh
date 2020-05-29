#BSUB -o out/hidost_similarity_job.%J
#BSUB -e out/hidost_similarity_job.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_similarity_job

cd ~/codnn
source env/bin/activate

python weight_similarity.py hidost -a custom_sigmoid -o similarity_out_final/hidost_similarity.json -n 64 -N 100000 -N 200000 -N 400000 -N 800000

