"""
Streamlit Web UI Application for Intelligent Document Understanding.
Provides an interactive dashboard for visual bounding boxes, KIE tables,
DocVQA chat, and JSON/CSV downloads.
"""

import streamlit as st
import json
import io
import os
from PIL import Image
import pandas as pd

from idu_core.pipeline import DocumentUnderstandingPipeline
from idu_core.config import config
from train.synthetic_generator import generate_sample_files

# Page configuration
st.set_page_config(
    page_title="Intelligent Document Understanding System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline(engine_name: str):
    return DocumentUnderstandingPipeline(ocr_engine=engine_name)

# Sidebar
st.sidebar.title("⚙️ IDU Configuration")

ocr_choice = st.sidebar.selectbox(
    "OCR Engine",
    options=["auto", "synthetic", "tesseract", "easyocr"],
    index=0,
    help="Select the OCR text recognition backend."
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Document Source")

# Ensure samples exist
samples_dir = config.samples_dir
if not os.path.exists(os.path.join(samples_dir, "sample_invoice.png")):
    generate_sample_files(samples_dir)

sample_options = ["None (Upload Custom Document)", "sample_invoice.png", "sample_receipt.png", "sample_medical_form.png"]
selected_sample = st.sidebar.selectbox("Choose Sample Document", sample_options)

uploaded_file = st.sidebar.file_uploader(
    "Or Upload Document (PDF, PNG, JPG)",
    type=["pdf", "png", "jpg", "jpeg", "tiff"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Hardware Device:** `{config.model.device.upper()}`")
st.sidebar.markdown(f"**System Version:** `v{config.version}`")

# Header
st.markdown('<div class="main-title">📄 Intelligent Document Understanding System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning Document AI: Layout Analysis, Key Information Extraction, Table Parsing & DocVQA</div>', unsafe_allow_html=True)

# Determine file to process
doc_bytes = None
doc_name = ""

if uploaded_file is not None:
    doc_bytes = uploaded_file.read()
    doc_name = uploaded_file.name
elif selected_sample != "None (Upload Custom Document)":
    sample_path = os.path.join(samples_dir, selected_sample)
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            doc_bytes = f.read()
        doc_name = selected_sample

if doc_bytes is None:
    st.info("👈 Please select a sample document or upload your own document from the sidebar to begin analysis.")
    st.stop()

# Execution
with st.spinner("Processing document through Deep Learning IDU Pipeline..."):
    pipeline = load_pipeline(ocr_choice)
    try:
        # Check if PDF or image
        if doc_name.lower().endswith(".pdf"):
            result = pipeline.process(doc_bytes, generate_visualizations=True)
        else:
            img = Image.open(io.BytesIO(doc_bytes))
            result = pipeline.process(img, generate_visualizations=True)
    except Exception as e:
        st.error(f"Error during document processing: {str(e)}")
        st.stop()

# Top Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Document Type", result.document_type, f"{int(result.type_confidence * 100)}% Conf")
with col2:
    st.metric("Total Pages", result.total_pages)
with col3:
    st.metric("Tokens Extracted", result.total_tokens)
with col4:
    st.metric("Entities Found", result.total_entities)
with col5:
    st.metric("Tables Detected", result.total_tables, f"{result.execution_time_seconds:.2f}s")

# Main Analysis Tabs
tab_vis, tab_kie, tab_tables, tab_vqa, tab_class = st.tabs([
    "🖼️ Visual Inspection",
    "🏷️ Key Information Extraction (KIE)",
    "📊 Table Parsing",
    "💬 DocVQA (Ask Questions)",
    "🎯 Document Classification"
])

# Tab 1: Visual Inspection
with tab_vis:
    st.subheader("Annotated Document with Bounding Boxes")
    page_nums = [p.page_num for p in result.pages]
    selected_page_idx = 0
    if len(page_nums) > 1:
        sel_page = st.selectbox("Select Page", page_nums)
        selected_page_idx = sel_page - 1

    page = result.pages[selected_page_idx]
    
    col_img, col_info = st.columns([2, 1])
    with col_img:
        if page.annotated_image:
            st.image(page.annotated_image, caption=f"Page {page.page_num} Annotations", use_container_width=True)
        else:
            st.warning("Visual overlay unavailable.")
            
    with col_info:
        st.markdown("#### OCR Tokens & Layout")
        st.markdown(f"**Dimensions:** {page.width} x {page.height} px")
        st.markdown(f"**Tokens on Page:** {len(page.tokens)}")
        st.markdown("#### Extracted Raw Text")
        st.text_area("OCR Text Stream", page.raw_text, height=350)

# Tab 2: Key Information Extraction
with tab_kie:
    st.subheader("Extracted Semantic Entities")
    filtered_entities = [
        e for p in result.pages for e in p.entities if e.confidence >= conf_threshold
    ]
    
    if filtered_entities:
        table_data = []
        for e in filtered_entities:
            table_data.append({
                "Entity Label": e.label,
                "Extracted Text": e.text,
                "Normalized Value": e.normalized_value,
                "Confidence": f"{int(e.confidence * 100)}%",
                "Bounding Box": f"[{e.box.x0}, {e.box.y0}, {e.box.x1}, {e.box.y1}]"
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.info("No entities found matching the confidence threshold.")

    # Spatial Question-Answer Pairs
    all_pairs = [p for page in result.pages for p in page.linked_pairs]
    if all_pairs:
        st.markdown("#### 🔗 Linked Question-Answer Pairs")
        pair_data = []
        for pair in all_pairs:
            pair_data.append({
                "Question": pair.question.text,
                "Linked Answer": pair.answer.text,
                "Answer Type": pair.answer.label,
                "Spatial Distance": f"{pair.spatial_distance:.1f} px"
            })
        st.dataframe(pd.DataFrame(pair_data), use_container_width=True)

    # Download JSON button
    json_str = json.dumps(result.to_dict(), indent=2)
    st.download_button(
        label="📥 Download Complete Extraction (JSON)",
        data=json_str,
        file_name=f"{doc_name}_extracted.json",
        mime="application/json"
    )

# Tab 3: Tables
with tab_tables:
    st.subheader("Extracted Tabular Structures")
    all_tables = [t for p in result.pages for t in p.tables]
    if all_tables:
        for idx, t in enumerate(all_tables, start=1):
            st.markdown(f"##### Table #{idx} (Rows: {len(t.rows)}, Cols: {len(t.headers)})")
            df = t.to_dataframe()
            st.dataframe(df, use_container_width=True)
            
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Download Table #{idx} as CSV",
                data=csv_data,
                file_name=f"table_{idx}.csv",
                mime="text/csv",
                key=f"dl_table_{idx}"
            )
    else:
        st.info("No tabular structures detected in this document.")

# Tab 4: DocVQA
with tab_vqa:
    st.subheader("Document Visual Question Answering")
    st.write("Ask natural language questions about the document's content, totals, dates, or vendor.")
    
    col_q, col_btn = st.columns([4, 1])
    with col_q:
        user_q = st.text_input(
            "Enter your question:",
            placeholder="e.g. What is the total amount due?"
        )
    with col_btn:
        st.write("")
        st.write("")
        ask_btn = st.button("Ask Model", type="primary")

    # Suggested prompt pills
    st.markdown("**Suggested Questions:**")
    pill_cols = st.columns(4)
    if pill_cols[0].button("Total Amount?"):
        user_q = "What is the total amount?"
        ask_btn = True
    if pill_cols[1].button("Invoice Date?"):
        user_q = "What is the invoice date?"
        ask_btn = True
    if pill_cols[2].button("Invoice Number?"):
        user_q = "What is the invoice number?"
        ask_btn = True
    if pill_cols[3].button("Vendor Name?"):
        user_q = "Who is the vendor?"
        ask_btn = True

    if (ask_btn or user_q) and user_q:
        qa_res = pipeline.ask(user_q, result)
        st.markdown("---")
        st.markdown(f"**Answer:** `{qa_res['answer']}`")
        st.markdown(f"**Confidence:** `{int(qa_res['confidence'] * 100)}%`")
        if qa_res.get("bounding_box"):
            st.markdown(f"**Coordinates:** `{qa_res['bounding_box']}`")

# Tab 5: Document Classification
with tab_class:
    st.subheader("Document Category Classification")
    st.markdown(f"Predicted Document Type: **{result.document_type}** ({int(result.type_confidence * 100)}% confidence)")
    
    prob_df = pd.DataFrame([
        {"Category": k, "Probability": v}
        for k, v in result.classification_probabilities.items()
    ]).sort_values(by="Probability", ascending=False)
    
    st.bar_chart(prob_df.set_index("Category"))
