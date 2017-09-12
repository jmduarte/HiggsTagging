# HiggsTagging
Credit to https://github.com/mstoye/DeepJet. Install Miniconda2 by sourcing `install_miniconda.sh` in your home directory. You may need to log out and log back in after this
```bash
cp install_miniconda.sh ~/
cd ~
source install_miniconda.sh
```

Then install the rest of the dependencies:
```bash
cd ~/HiggsTagging
source install.sh
```

Finally each time you log in set things up:
```bash
source setup.sh
```

To run a simple training:
```bash
cd ~/HiggsTagging/train
python train_deepdoubleb_simple.py
```
and evaluate the training:
```bash
python eval_deepdoubleb_simple.py
```
