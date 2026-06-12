import os
import numpy as np
import streamlit as st
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from PIL import Image

# Folder where uploaded photos are saved
UPLOAD_FOLDER = "uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_and_embed_images(folder):
    embeddings = []
    image_paths = []
    
    files = [f for f in os.listdir(folder) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    progress = st.progress(0)
    
    for i, filename in enumerate(files):
        path = os.path.join(folder, filename)
        try:
            result = DeepFace.represent(
                img_path=path,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='mtcnn'
            )
            if result:
                embedding = result[0]['embedding']
                embeddings.append(embedding)
                image_paths.append(path)
        except Exception as e:
            st.warning(f"⚠️ Skipped {filename}: {e}")
        
        progress.progress((i + 1) / len(files))
    
    return embeddings, image_paths


def cluster_embeddings(embeddings):
    embeddings_array = np.array(embeddings)
    
    clustering = DBSCAN(
        eps=13,
        min_samples=2,
        metric='euclidean'
    ).fit(embeddings_array)
    
    return clustering.labels_


def display_clusters(labels, image_paths):
    clusters = {}
    for label, path in zip(labels, image_paths):
        clusters.setdefault(label, []).append(path)

    # Separate real clusters from noise
    real_clusters = {k: v for k, v in clusters.items() if k != -1}
    noise = clusters.get(-1, [])
    
    if not real_clusters:
        st.error("❌ Could not group any photos! Try uploading more photos of same person.")
        return
    
    st.success(f"✅ Found {len(real_clusters)} people in your photos!")
    st.balloons()
    
    for label, paths in sorted(real_clusters.items()):
        st.markdown(f"### 👤 Person {label + 1}")
        st.caption(f"{len(paths)} photo(s)")
        
        cols = st.columns(min(len(paths), 5))
        for i, path in enumerate(paths):
            with cols[i % 5]:
                st.image(path, width=150,
                        caption=os.path.basename(path))
        st.divider()
    
    # Show ungrouped photos separately
    if noise:
        st.markdown("### 🔍 Unrecognized Photos")
        st.caption("These photos couldn't be grouped")
        cols = st.columns(min(len(noise), 5))
        for i, path in enumerate(noise):
            with cols[i % 5]:
                st.image(path, width=150,
                        caption=os.path.basename(path))


# ─── STREAMLIT UI ───────────────────────────

st.set_page_config(
    page_title="AutoCluster",
    page_icon="👤",
    layout="wide"
)

st.title("👤 AutoCluster")
st.subheader("AI Powered Photo Clustering Using Face Recognition")
st.write("Upload photos → Click Cluster → See people grouped automatically!")

st.markdown("---")

uploaded_files = st.file_uploader(
    "📁 Upload Photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    # Clear old photos
    for old_file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, old_file))
    
    # Save new photos
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ {len(uploaded_files)} photos ready!")

    if st.button("🔍 Cluster Faces", type="primary"):
        
        with st.spinner("🧠 Detecting faces..."):
            embeddings, image_paths = load_and_embed_images(UPLOAD_FOLDER)
        
        if not embeddings:
            st.error("❌ No faces detected! Please upload clear photos.")
        else:
            st.info(f"🎯 Detected {len(embeddings)} faces — clustering now...")
            labels = cluster_embeddings(embeddings)
            display_clusters(labels, image_paths)