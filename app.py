"""Station Safety Inspector - Streamlit app using your trained YOLO model.
Run:  streamlit run app.py
"""
import os
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

REQUIRED = ["OxygenTank", "NitrogenTank", "FirstAidBox", "FireAlarm",
            "SafetySwitchPanel", "EmergencyPhone", "FireExtinguisher"]
norm = lambda s: s.replace(" ", "").replace("_", "").lower()

st.set_page_config(page_title="Station Safety Inspector", page_icon="🛰️", layout="wide")
st.title("🛰️ Station Safety Inspector")
st.caption("Upload a space station image. The YOLO model checks that all 7 safety objects are present.")

# ---- sidebar ----
model_path = st.sidebar.text_input("Model weights", "best.pt")
conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
if "runs" not in st.session_state:
    st.session_state.runs = 0

@st.cache_resource
def load_model(path):
    return YOLO(path)

if not os.path.exists(model_path):
    st.error(f"'{model_path}' not found. Copy your trained weights "
             "(runs/detect/train/weights/best.pt) next to app.py.")
    st.stop()

model = load_model(model_path)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    result = model.predict(np.array(img), conf=conf, verbose=False)[0]
    st.session_state.runs += 1

    found = {}
    for box in result.boxes:
        name = model.names[int(box.cls)]
        found[norm(name)] = max(found.get(norm(name), 0), float(box.conf))

    left, right = st.columns([2, 1])
    left.image(result.plot()[:, :, ::-1], caption="Detections", use_container_width=True)

    with right:
        st.subheader("Inspection report")
        missing = []
        for item in REQUIRED:
            if norm(item) in found:
                st.success(f"{item}: found ({found[norm(item)]:.2f})")
            else:
                st.error(f"{item}: MISSING")
                missing.append(item)
        if missing:
            st.warning("Not safe. Missing: " + ", ".join(missing))
        else:
            st.balloons()
            st.info("All 7 safety objects detected.")
        st.metric("Inspections run", st.session_state.runs)

with st.expander("How Falcon keeps this model up to date"):
    st.markdown("""
1. Flag low-confidence or missed detections from the field.
2. Recreate that object, lighting or angle in the Falcon digital twin.
3. Generate new labeled synthetic images automatically.
4. Retrain YOLO, compare mAP@0.5 with the baseline, redeploy if better.
""")
