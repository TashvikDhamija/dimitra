python3 -m venv Dimitra_venv
pip install -r requirements.txt
cd Deep3DFaceRecon_pytorch/nvdiffrast
pip install .
cd ../../
mv utils.py Dimitra_venv/lib/python3.12/site-packages/realesrgan/
mv degradations.py Dimitra_venv/lib/python3.12/site-packages/basicsr/data/
