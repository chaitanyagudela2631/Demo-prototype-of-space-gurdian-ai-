# Station Safety Inspector

Detects 7 safety objects in space station images (Duality AI Falcon dataset) and reports missing ones.

## Run
1. Train with the hackathon scripts: `python train.py` (inside the EDU conda env).
2. Copy `runs/detect/train/weights/best.pt` into this folder.
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

`demo_prototype.html` is a no-install visual demo (double-click to open, simulated detections).
