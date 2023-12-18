module load cray-python
module load rocm/5.4.0
python3 -m venv /path/to/newenv --system-site-packages
source /path/to/newenv/bin/activate
pip3 install --user tensorflow-rocm --upgrade
pip3 install tensorflow
pip3 install SmilesPE
pip3 install transformers
./submit_frontier.sh
