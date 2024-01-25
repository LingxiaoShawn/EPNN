conda create -n epnn python=3.10
conda activate epnn

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

conda install openbabel fsspec rdkit -c conda-forge

pip install ogb
pip install wandb
pip install yacs
pip install opt_einsum
pip install pytorch-lightning # required by graphgym 
pip install typing_extensions
pip install torchmetrics==0.9.1
pip install tensorboardX



python main.py --cfg configs/GRIT/peptides-func-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:6"  dataset.dir 'datasets'
python main.py --cfg configs/GRIT/peptides-func-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:6"  dataset.dir 'datasets'
python main.py --cfg configs/GRIT/peptides-func-GRIT-RRWP.yaml accelerator "cuda:7"  dataset.dir 'datasets'
python main.py --cfg configs/GRIT/peptides-struct-GRIT-RRWP.yaml accelerator "cuda:4"  dataset.dir 'datasets'

python main.py --cfg configs/GRIT/cifar10-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:7"  dataset.dir 'datasets'
python main.py --cfg configs/GRIT/cluster-GRIT-RRWP.yaml accelerator "cuda:6"  dataset.dir 'datasets'
python main.py --cfg configs/GRIT/cluster-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:5"  dataset.dir 'datasets'

# python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:5" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag fExp_gMLP_concat
# python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:6" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag fExp_gMLP_add
# python main.py --cfg configs/GRIT/zinc-GRIT-RRWP.yaml  accelerator "cuda:7" optim.max_epoch 2000 seed 41 dataset.dir '.datasets' name_tag original

python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:7" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag f512Exp_g256Exp
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:6" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag f512ExpSiLU_g256MLP
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:5" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag f512MLP_g256MLP


python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:4" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag f512Exp_g256MLPsilu_add
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:5" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag f512Exp_g256MLP_add
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:6" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag f512Exp_g256MLP_concat
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:7" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag f512Exp_g256MLPsilu_concat

python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:6" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag f512Exp_g256MLPsilu_add_doubleinc
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:6" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag f512Exp_g256Sigmoid_add
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:7" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag f512Exp_g256MLPact
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP-EigenBasis.yaml accelerator "cuda:4" optim.max_epoch 2000 seed 41 dataset.dir 'datasets' name_tag f512IncExp_g256Inc