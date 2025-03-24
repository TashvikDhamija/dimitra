# AI Killed the Video Star

[Paper](http://arxiv.org/abs/2502.17198) | [Project Page](https://tashvikdhamija.github.io/dimitra/)

![Project Image](static/images/model_diagram1.png)


---

## Installation

Follow these instruction to install the code. It will require a NVIDIA gpu with more than 8 Go of memory.

```bash
git clone https://github.com/TashvikDhamija/dimitra.git
cd dimitra
pip install -r requirements.txt
cd Deep3DFaceRecon_pytorch
git clone -b 0.3.0 https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
cd ../../

