import streamlit as st

# Set page config at the very beginning
st.set_page_config(page_title="Advanced Contract Analysis App", layout="wide")

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
import spacy
from spacy.tokens import Span, SpanGroup
from PyPDF2 import PdfReader
import docx
import io

# Load models and resources
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    contract_model = AutoModelForSequenceClassification.from_pretrained('legalbert_contract_classifier')
    contract_model.to(device)
    contract_tokenizer = AutoTokenizer.from_pretrained('legalbert_contract_classifier')
    contract_label_encoder = joblib.load('label_encoder.pkl')
    nlp = spacy.load("en_core_web_lg")
    return contract_model, contract_tokenizer, contract_label_encoder, nlp, device

contract_model, contract_tokenizer, contract_label_encoder, nlp, device = load_models()

# Set up span ruler for clause detection
if "span_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("span_ruler")
else:
    ruler = nlp.get_pipe("span_ruler")

patterns = [
    {"label": "Governing Law", "pattern": "the laws of the"},
    {"label": "Governing Law", "pattern": "shall be governed by"},
    {"label": "Governing Law", "pattern": "governed in accordance with"},
    {"label": "Governing Law", "pattern": "in accordance with the laws of"},
    {"label": "Governing Law", "pattern": "This Agreement is subject to"},
    {"label": "Termination for Convenience", "pattern": "may terminate this Agreement"},
    {"label": "Termination for Convenience", "pattern": "written notice to the"},
    {"label": "Termination for Convenience", "pattern": "prior written notice to"},
    {"label": "Termination for Convenience", "pattern": "this Agreement at any time prior to"},
    {"label": "Termination for Convenience", "pattern": "to terminate this Agreement"}
]
ruler.add_patterns(patterns)

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_word(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def predict_contract_type(text):
    inputs = contract_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = contract_model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_label = contract_label_encoder.inverse_transform([predicted_class])[0]
    
    return predicted_label

def detect_and_classify_clauses(text):
    doc = nlp(text)
    doc.spans["test"] = SpanGroup(doc)

    for sentence in doc.sents:
        for span in doc.spans["ruler"]:
            if span.start >= sentence.start and span.end <= sentence.end:
                new_span = Span(doc, start=sentence.start, end=sentence.end, label=span.label_)
                doc.spans["test"].append(new_span)
                doc.set_ents(entities=[new_span], default="unmodified")

    clauses = [(span.text, span.label_) for span in doc.spans["test"]]
    df = pd.DataFrame(clauses, columns=["Clause Content", "Clause Type"])
    
    # Remove duplicate clauses
    df = df.drop_duplicates(subset=["Clause Content"])
    
    return df

# Streamlit UI
st.title("📄 Advanced Contract Analysis App")
st.subheader("Upload a contract (PDF or Word) to detect the contract type and specific clauses using AI.")

uploaded_file = st.file_uploader("Choose a PDF or Word file", type=["pdf", "docx"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    with st.spinner("Analyzing the document..."):
        if file_extension == 'pdf':
            extracted_text = extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            extracted_text = extract_text_from_word(uploaded_file)
        else:
            st.error("Unsupported file type!")

        # Display extracted text preview
        with st.expander("📄 Extracted Text Preview"):
            st.write(extracted_text[:500] + '...')

        # Predict contract type
        contract_type = predict_contract_type(extracted_text)
        st.markdown(f"### Contract Type: **:blue[{contract_type}]**")

        # Detect and display clauses
        clause_df = detect_and_classify_clauses(extracted_text)

        st.subheader("Detected Clauses")
        if not clause_df.empty:
            grouped_clauses = clause_df.groupby("Clause Type")

            for clause_type, group in grouped_clauses:
                with st.expander(f"📌 {clause_type} Clauses ({len(group)})"):
                    for i, (_, row) in enumerate(group.iterrows(), 1):
                        st.markdown(f"**Clause {i}:**")
                        st.write(row["Clause Content"])
                        st.markdown("---")
        else:
            st.write("No meaningful clauses were detected in this contract.")

    st.success("Analysis complete!")

    # Add a download button for the analysis results
    analysis_results = f"Contract Type: {contract_type}\n\n"
    for clause_type, group in grouped_clauses:
        analysis_results += f"{clause_type} Clauses:\n"
        for i, (_, row) in enumerate(group.iterrows(), 1):
            analysis_results += f"Clause {i}: {row['Clause Content']}\n\n"
    
    st.download_button(
        label="Download Analysis Results",
        data=analysis_results,
        file_name="contract_analysis_results.txt",
        mime="text/plain"
    )

# Sidebar with app information
st.sidebar.header("About this app")
st.sidebar.info("""
This advanced app uses a fine-tuned LegalBERT model to classify legal contracts and a custom clause detection model to identify specific clauses with improved accuracy and meaningfulness.
""")

st.sidebar.header("App Features")
st.sidebar.markdown("""
- 📂 Upload PDF or Word files
- 🧠 AI-based contract type detection
- 🔍 Advanced clause detection using custom model
- 🏷️ Clause type classification
- 📊 Interactive and expandable results display
- 💾 Download analysis results
""")

st.sidebar.header("How to use")
st.sidebar.markdown("""
1. Upload a PDF or Word document containing a legal contract.
2. The app will automatically extract the text and analyze it.
3. View the predicted contract type and detected clauses.
4. Expand each clause type to see the specific clauses identified.
5. Download the analysis results for further review.
""")