### How to create environment
```
module load cray-python
module load rocm/5.4.0
python3 -m venv /path/to/newenv --system-site-packages
source /path/to/newenv/bin/activate
pip3 install --user tensorflow-rocm --upgrade
pip3 install tensorflow
pip3 install SmilesPE
pip3 install transformers
```
### How to run
Modify ``submit_frontier.sh``

    Change ``/lustre/orion/chm155/scratch/avasan/ST_Code/Benchmarks/Pilot1/ST1/Inf_STRev_MultiNode_Frontier`` to your working path

    Change ``/autofs/nccs-svm1_proj/chm155/avasan/envs/st_env/bin/activate`` to your environment path

``./submit_frontier.sh``
