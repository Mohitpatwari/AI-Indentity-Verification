import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import google.generativeai as genai
from paddleocr import PaddleOCR
import insightface
from insightface.app import FaceAnalysis

# Page Config
st.set_page_config(
    page_title="AI Identity Verification Platform",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ AI Identity Verification Platform")
st.write("Upload your ID document and a selfie to run automated KYC, OCR, and Face-Matching analysis.")

# Configure Gemini API from Streamlit Secrets or environment
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if api_key:
    genai.configure(api_key=api_key)

# Load Heavy Models Cached in Memory
@st.cache_resource
def load_ocr_model():
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

@st.cache_resource
def load_face_analyzer():
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

with st.spinner("Loading AI models (PaddleOCR & InsightFace)..."):
    ocr = load_ocr_model()
    face_app = load_face_analyzer()

# Sidebar Status
with st.sidebar:
    st.header("System Status")
    if api_key:
        st.success("Gemini API Key Loaded ✅")
    else:
        st.error("Gemini API Key Missing ❌")
    st.info("Models initialized via Streamlit cache.")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    id_file = st.file_uploader("Upload ID Document (PAN/Aadhaar)", type=["jpg", "jpeg", "png"])
with col2:
    selfie_file = st.file_uploader("Upload Selfie", type=["jpg", "jpeg", "png"])

if id_file and selfie_file:
    id_img = Image.open(id_file)
    selfie_img = Image.open(selfie_file)
    
    st.image([id_img, selfie_img], caption=["ID Document", "Selfie"], width=300)
    
    if st.button("Run Verification Pipeline", type="primary"):
        with st.spinner("Executing OCR Extraction, Face Embeddings & Gemini Fraud Analysis..."):
            
            # Convert images to OpenCV formats
            id_cv = cv2.cvtColor(np.array(id_img), cv2.COLOR_RGB2BGR)
            selfie_cv = cv2.cvtColor(np.array(selfie_img), cv2.COLOR_RGB2BGR)
            
            # 1. PaddleOCR Text Extraction
            ocr_result = ocr.ocr(id_cv, cls=True)
            extracted_texts = []
            if ocr_result and ocr_result[0]:
                extracted_texts = [line[1][0] for line in ocr_result[0]]
            full_ocr_text = "\n".join(extracted_texts)
            
            # 2. InsightFace Face Embedding & Cosine Similarity Match
            id_faces = face_app.get(id_cv)
            selfie_faces = face_app.get(selfie_cv)
            
            similarity_score = 0.0
            face_match_status = "Failed"
            
            if len(id_faces) > 0 and len(selfie_faces) > 0:
                id_embedding = id_faces[0].embedding
                selfie_embedding = selfie_faces[0].embedding
                
                # Compute Cosine Similarity
                similarity_score = float(np.dot(id_embedding, selfie_embedding) / (
                    np.linalg.norm(id_embedding) * np.linalg.norm(selfie_embedding)
                ))
                # Threshold check (e.g., 0.45)
                if similarity_score >= 0.45:
                    face_match_status = "Matched ✅"
                else:
                    face_match_status = "Mismatch ❌"
            else:
                face_match_status = "Face not detected clearly in one or both images."

            # 3. Gemini Analysis Integration
            gemini_report = "Gemini client not initialized."
            if api_key:
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = f"""
                    You are an expert KYC and Fraud Analytics compliance agent. Analyze the following extracted OCR text from an identity document and the biometric matching results.
                    
                    Extracted OCR Data:
                    {full_ocr_text}
                    
                    Face Matching Results:
                    - Status: {face_match_status}
                    - Cosine Similarity Score: {similarity_score:.4f}
                    
                    Provide a concise JSON-like compliance breakdown containing:
                    1. Document Authenticity assessment
                    2. Data consistency review
                    3. Final Recommendation (Approved / Rejected / Manual Review)
                    """
                    response = model.generate_content(prompt)
                    gemini_report = response.text
                except Exception as e:
                    gemini_report = f"Error communicating with Gemini API: {str(e)}"

            # Render Results Tabs
            st.markdown("---")
            st.subheader("📋 Verification & Comparison Results")
            
            tab1, tab2, tab3 = st.tabs(["Biometric Face Match", "OCR Text Extraction", "AI Summary Report"])
            
            with tab1:
                st.metric(label="Face Similarity Score", value=f"{similarity_score:.4f}", delta=face_match_status)
                if len(id_faces) == 0:
                    st.warning("No face detected in the ID document.")
                if len(selfie_faces) == 0:
                    st.warning("No face detected in the selfie image.")
                
            with tab2:
                st.text_area("Raw Extracted Text from Document", value=full_ocr_text if full_ocr_text else "No text found.", height=200)
                
            with tab3:
                st.markdown(gemini_report)
                
            st.success("Verification Pipeline Completed Successfully!")