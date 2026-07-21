So i have to learn jax

it's quite hard ngl

pip install -r requirements.txt

python run.py                 # synthetic protein
python run.py --mode contact  # contact map only (masked, hard)
python run.py --pdb 1UBQ --chain A     # real ubiquitin backbone, it should be able to fetch it 
python run.py --pdb 1UBQ --chain A --steps 500 