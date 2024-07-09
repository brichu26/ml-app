import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to generate summary
def generate_summary(dialogue, model, tokenizer):
    inputs = tokenizer(dialogue, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with actual model path if different
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app UI
st.title('🎈 Medical Summarization of Doctor-Patient Dialogues')

st.write('Enter the doctor-patient dialogue below to generate a medical summary.')

# Text input box for the dialogue
dialogue = st.text_area("Doctor-Patient Dialogue", height=300)

if st.button("Generate Summary"):
    if dialogue:
        with st.spinner('Generating summary...'):
            summary = generate_summary(dialogue, model, tokenizer)
        st.write("### Generated Summary")
        st.write(summary)
    else:
        st.write("Please enter a doctor-patient dialogue to generate a summary.")
