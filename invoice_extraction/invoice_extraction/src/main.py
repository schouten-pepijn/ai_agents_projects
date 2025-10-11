"""streamlit run src/main.py"""
import logging
import json
import pandas as pd
from datetime import datetime
import streamlit as st
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from core.dockling_extractor import DoclingRAGExtractor
from core.vectorstore_in_mem import VectorStoreInMem
from core.tools import ToolRegistry
from core.schema import field_schema
from configs.llm_settings import get_ollama_chat_client, get_ollama_embed_client

logging.root.setLevel(logging.INFO)
logger = logging.getLogger()

st.set_page_config(page_title="Invoice Extraction with LLM", layout="wide", page_icon="📄")

st.markdown("""
<style>
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px;
        margin: 5px 0;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 5px 0;
    }
    .error-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px;
        margin: 5px 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("📄 Invoice Extraction Application")
st.markdown("Extract structured data from invoices using AI-powered RAG & Docling")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
top_k = st.sidebar.slider("Top K chunks for retrieval", min_value=1, max_value=15, value=5, step=1)
max_tokens = st.sidebar.select_slider("Max tokens per chunk", options=[64, 128, 256, 512, 1024, 2048], value=256)
show_debug = st.sidebar.checkbox("Show debug information", value=False)

st.sidebar.divider()
st.sidebar.header("📊 Extraction Fields")
st.sidebar.markdown(f"**{len(field_schema)}** fields will be extracted:")
for field_name in field_schema.keys():
    st.sidebar.markdown(f"- {field_name.replace('_', ' ').title()}")

# Main UI
uploaded_file = st.file_uploader(
    "Upload invoice (PDF/Image)", 
    type=["pdf", "png", "jpg", "jpeg"], 
    accept_multiple_files=False,
    help="Upload a PDF or image file of an invoice"
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    run = st.button("🚀 Run Extraction", type="primary", use_container_width=True)
with col2:
    clear = st.button("🔄 Clear Results", use_container_width=True)

if clear:
    if 'extraction_results' in st.session_state:
        del st.session_state['extraction_results']
    if 'chunks' in st.session_state:
        del st.session_state['chunks']
    st.rerun()

if uploaded_file and run:
    try:
        with st.spinner("🔄 Initializing components..."):
            # Initialize clients
            llm_client = get_ollama_chat_client()
            emb_client = get_ollama_embed_client()
            
            # Initialize components
            vector_store = VectorStoreInMem(emb_model=emb_client)
            tools = ToolRegistry()
            converter = DocumentConverter()
            chunker = HybridChunker(
                tokenizer="sentence-transformers/all-MiniLM-L6-v2",
                max_tokens=max_tokens,
                merge_peers=True
            )
            
            parser = DoclingRAGExtractor(
                llm_client=llm_client,
                emb_client=emb_client,
                converter=converter,
                vector_store=vector_store,
                tools=tools,
                chunker=chunker,
                max_chunk_tokens=max_tokens,
            )
        
        with st.spinner(f"📖 Processing document: {uploaded_file.name}..."):
            chunks = parser.process_document(uploaded_file.getvalue())
            logger.info(f"Parsed {len(chunks)} chunks")
            
            # Add chunks to vector store
            vector_store.add_chunks(chunks)
            st.session_state['chunks'] = chunks
            
        st.success(f"✅ Document processed! Created {len(chunks)} text chunks")
        
        # Progress bar for extraction
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        extracted_data = {}
        total_fields = len(field_schema)
        
        for idx, (field_name, field_config) in enumerate(field_schema.items()):
            progress = (idx + 1) / total_fields
            progress_bar.progress(progress)
            status_text.text(f"Extracting: {field_name.replace('_', ' ').title()} ({idx + 1}/{total_fields})")
            
            # Extract single field
            single_result = parser.extract_fields(
                field_schema={field_name: field_config},
                top_k=top_k
            )
            extracted_data.update(single_result)
        
        progress_bar.empty()
        status_text.empty()
        
        st.session_state['extraction_results'] = extracted_data
        
        st.success("✨ Extraction complete!")
        
    except Exception as e:
        st.error(f"❌ Error during extraction: {str(e)}")
        logger.error(f"Extraction error: {e}", exc_info=True)
        if show_debug:
            st.exception(e)

# Display results if available
if 'extraction_results' in st.session_state:
    extracted_data = st.session_state['extraction_results']
    
    st.header("✅ Extracted Invoice Data")
    
    # Calculate summary statistics
    total_fields = len(extracted_data)
    successful = sum(1 for v in extracted_data.values() if v.get('status') == 'success')
    not_found = sum(1 for v in extracted_data.values() if v.get('status') == 'not_found')
    low_conf = sum(1 for v in extracted_data.values() if v.get('status') == 'low_confidence')
    errors = sum(1 for v in extracted_data.values() if v.get('status') == 'error')
    avg_confidence = sum(v.get('confidence', 0) for v in extracted_data.values()) / total_fields if total_fields > 0 else 0
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Fields", total_fields)
    col2.metric("✅ Extracted", successful, delta=f"{successful/total_fields*100:.0f}%")
    col3.metric("⚠️ Low Confidence", low_conf)
    col4.metric("❌ Not Found", not_found)
    col5.metric("📊 Avg Confidence", f"{avg_confidence:.0%}")
    
    st.divider()
    
    # Extracted fields display
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📋 Extracted Fields")
        
        for field_name, field_data in extracted_data.items():
            value = field_data.get('value')
            confidence = field_data.get('confidence', 0)
            status = field_data.get('status', 'unknown')
            
            # Determine confidence class
            if confidence >= 0.7:
                conf_class = "confidence-high"
                conf_label = "High"
            elif confidence >= 0.4:
                conf_class = "confidence-medium"
                conf_label = "Medium"
            else:
                conf_class = "confidence-low"
                conf_label = "Low"
            
            # Display field
            with st.container():
                field_label = field_name.replace('_', ' ').title()
                
                if status == 'success':
                    st.markdown(f"**{field_label}:** {value}")
                    st.markdown(f"<span class='{conf_class}'>Confidence: {confidence:.0%} ({conf_label})</span>", unsafe_allow_html=True)
                elif status == 'low_confidence':
                    st.markdown(f"**{field_label}:** {value} ⚠️")
                    st.markdown(f"<span class='{conf_class}'>Confidence: {confidence:.0%} (Low - verify manually)</span>", unsafe_allow_html=True)
                elif status == 'not_found':
                    st.markdown(f"**{field_label}:** _Not found_")
                elif status == 'error':
                    st.markdown(f"**{field_label}:** ❌ Error")
                    if show_debug and 'error' in field_data:
                        st.caption(f"Error: {field_data['error']}")
                
                if show_debug and field_data.get('source'):
                    with st.expander(f"Show source for {field_label}"):
                        st.code(field_data['source'], language=None)
                
                st.divider()
    
    with col2:
        st.subheader("📤 Export Options")
        
        # Prepare export data
        export_simple = {k: v.get('value') for k, v in extracted_data.items()}
        export_detailed = extracted_data
        
        # JSON export
        st.download_button(
            label="📥 Download as JSON",
            data=json.dumps(export_simple, indent=2),
            file_name=f"invoice_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        # CSV export
        df = pd.DataFrame([
            {
                'Field': k.replace('_', ' ').title(),
                'Value': v.get('value', ''),
                'Confidence': f"{v.get('confidence', 0):.0%}",
                'Status': v.get('status', '')
            }
            for k, v in extracted_data.items()
        ])
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"invoice_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Detailed JSON export
        st.download_button(
            label="📥 Download Detailed JSON",
            data=json.dumps(export_detailed, indent=2),
            file_name=f"invoice_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            help="Includes confidence scores and source information"
        )
        
        st.divider()
        
        # Quality indicators
        st.subheader("📊 Quality Indicators")
        
        if avg_confidence >= 0.7:
            st.success("✅ High quality extraction")
        elif avg_confidence >= 0.4:
            st.warning("⚠️ Medium quality - verify results")
        else:
            st.error("❌ Low quality - manual review needed")
        
        st.metric("Fields with High Confidence", 
                 sum(1 for v in extracted_data.values() if v.get('confidence', 0) >= 0.7))
        st.metric("Fields Needing Review", 
                 sum(1 for v in extracted_data.values() if v.get('confidence', 0) < 0.7))
    
    # Document chunks visualization
    if show_debug and 'chunks' in st.session_state:
        st.header("🔍 Document Chunks Analysis")
        chunks = st.session_state['chunks']
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Total Chunks", len(chunks))
            st.metric("Avg Chunk Length", f"{sum(len(c['text']) for c in chunks) / len(chunks):.0f} chars")
        
        with st.expander("📄 View All Chunks"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i + 1}**")
                st.code(chunk['text'], language=None)
                st.caption(f"Metadata: {chunk['metadata']}")
                st.divider()

else:
    # Welcome message
    st.info("👆 Upload an invoice to get started!")
    
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        This professional invoice extraction system uses:
        
        1. **Docling** - Advanced document parsing and chunking
        2. **RAG (Retrieval-Augmented Generation)** - Semantic search for relevant context
        3. **LLM with Tools** - Intelligent extraction with specialized tools
        4. **Confidence Scoring** - Quality assessment for each extracted field
        
        **Features:**
        - Extracts 10+ common invoice fields
        - Provides confidence scores for each extraction
        - Supports multiple export formats (JSON, CSV, PDF)
        """)
