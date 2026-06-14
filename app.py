import os
import numpy as np
import streamlit as st
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from PIL import Image

UPLOAD_FOLDER = "uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_and_embed_images(folder):
    embeddings = []
    image_paths = []
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    progress = st.progress(0)
    status = st.empty()

    for i, filename in enumerate(files):
        path = os.path.join(folder, filename)
        status.markdown(f"Analyzing **{filename}**...")
        try:
            result = DeepFace.represent(
                img_path=path,
                model_name='Facenet512',
                enforce_detection=False,
                detector_backend='mtcnn'
            )
            if result:
                embeddings.append(result[0]['embedding'])
                image_paths.append(path)
        except Exception:
            pass
        progress.progress((i + 1) / len(files))

    status.empty()
    progress.empty()
    return embeddings, image_paths


def cluster_embeddings(embeddings):
    embeddings_array = np.array(embeddings)
    embeddings_array = normalize(embeddings_array)

    clustering = DBSCAN(
        eps=0.35,
        min_samples=2,
        metric='cosine'
    ).fit(embeddings_array)

    return clustering.labels_


def display_clusters(labels, image_paths):
    clusters = {}
    for label, path in zip(labels, image_paths):
        clusters.setdefault(label, []).append(path)

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
                img = Image.open(path)
                st.image(img,
                        width=150,
                        caption=os.path.basename(path))
        st.divider()

    if noise:
        st.markdown("### 🔍 Unrecognized Photos")
        st.caption("Blurry, black & white, or unique photos")
        cols = st.columns(min(len(noise), 5))
        for i, path in enumerate(noise):
            with cols[i % 5]:
                st.image(path,
                        width=150,
                        caption=os.path.basename(path))


# ─── UI ─────────────────────────────────────
st.set_page_config(
    page_title="AutoCluster",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    for old_file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, old_file))

    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"✅ {len(uploaded_files)} photos ready!")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        cluster_btn = st.button(
            "🔍 Cluster Faces",
            use_container_width=True
        )

    if cluster_btn:
        with st.spinner("🧠 Detecting faces..."):
            embeddings, image_paths = load_and_embed_images(UPLOAD_FOLDER)

        if not embeddings:
            st.error("❌ No faces detected!")
        else:
            st.info(f"🎯 Detected {len(embeddings)} faces — clustering now...")
            labels = cluster_embeddings(embeddings)
            display_clusters(labels, image_paths)