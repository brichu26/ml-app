import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load models and tokenizers
@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

# Dictionary to map model names to Hugging Face model identifiers
model_dict = {
    'Llama-3-8B-Instruct': 'google/llama-3-8b',  # Replace with actual model path
    'T5-Small': 't5-small',
    'Fine-Tuned BART': 'facebook/bart-large-cnn'  # Replace with actual model path if different
}

# Streamlit app UI
st.title('🎈 Medical Summarization of Doctor-Patient Dialogues 🤒')

st.write('Enter the doctor-patient dialogue below to generate a medical summary.')

# Text input box for the dialogue
dialogue = st.text_area("Doctor-Patient Dialogue", height=300)

# Model selection
model_choice = st.radio("Choose a model for summarization:", ('Llama-3-8B-Instruct(Base)', 'T5-Small', 'Fine-Tuned BART', 'Fine-Tuned Llama-3-8B'))

# Placeholder for the summary
summary_placeholder = st.empty()

if st.button("Generate Summary"):
    if dialogue:
        with st.spinner('Generating summary...'):
            # Load the selected model and tokenizer
            model_name = model_dict[model_choice]
            model, tokenizer = load_model_and_tokenizer(model_name)
            
            # Generate summary
            inputs = tokenizer(dialogue, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Display summary
            summary_placeholder.text_area("Generated Summary", value=summary, height=300)
    else:
        st.write("Please enter a doctor-patient dialogue to generate a summary.")
