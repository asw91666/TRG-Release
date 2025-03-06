# Environments
We train and test on the below setting.
CUDA==11.1
Python==3.8.17
torch==1.10.1

```bash
ENV_NAME="trg"
conda create --name $ENV_NAME python=3.8
conda activate $ENV_NAME
```

# Requirements
Install torch following the instructions below, or you can choose your version according to your CUDA.

```bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Install other requirements.

```bash
pip install -r requirements.txt
```

## torchgeometry library
Please follow below before run the code:
you need to change L301~L304 of `anaconda3/lib/python3.8/site-packages/torchgeometry/core/conversion.py` to below.

```bash
mask_c0 = mask_d2.float() * mask_d0_d1.float()
mask_c1 = mask_d2.float() * (1 - mask_d0_d1.float())
mask_c2 = (1 - mask_d2.float()) * mask_d0_nd1.float()
mask_c3 = (1 - mask_d2.float()) * (1 - mask_d0_nd1.float())
```

