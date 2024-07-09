import streamlit as st
from transformers import pipeline

# Function to load the summarization model
@st.cache_resource
def load_model(model_name):
    try:
        summarizer = pipeline("summarization", model=model_name)
        return summarizer
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

# Function to summarize dialogue
def summarize_text(dialogue, model_name):
    summarizer = load_model(model_name)
    if summarizer:
        try:
            summary = summarizer(dialogue, max_length=150, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            return None
    else:
        return None

# Set up the Streamlit app layout
st.title("Doctor-Patient Dialogue Summarizer")

# Input field for the dialogue
user_dialogue = st.text_area("Enter the doctor-patient dialogue here:", height=300)

# Model selection
model_choice = st.selectbox(
    "Select the summarization model:",
    ["Llama-3-8B", "T5"]
)

# Map user-friendly model names to actual model identifiers
model_mapping = {
    "Llama-3-8B": "knkarthick/MEETING_SUMMARY",  # Replace with the actual model name if different
    "T5": "t5-base"
}

# Submit button
if st.button("Generate Summary"):
    if user_dialogue:
        with st.spinner("Generating summary..."):
            selected_model = model_mapping[model_choice]
            summary = summarize_text(user_dialogue, selected_model)
            if summary:
                st.write("### Summary")
                st.write(summary)
            else:
                st.error("Failed to generate summary.")
    else:
        st.warning("Please enter a doctor-patient dialogue.")

# Display example dialogues and summaries for testing
if st.checkbox("Show example dialogues"):
    examples = {
        "Dialogue 1": "Doctor: How have you been feeling lately? Patient: I've been having a lot of headaches.",
        "Dialogue 2": "Doctor: Are you experiencing any chest pain? Patient: Yes, it gets worse when I exercise."
    }
    example_choice = st.selectbox("Choose an example dialogue:", list(examples.keys()))
    if example_choice:
        st.write("### Selected Dialogue")
        st.write(examples[example_choice])
        if st.button("Generate Summary for Example"):
            with st.spinner("Generating summary..."):
                selected_model = model_mapping[model_choice]
                summary = summarize_text(examples[example_choice], selected_model)
                if summary:
                    st.write("### Summary")
                    st.write(summary)
                else:
                    st.error("Failed to generate summary.")

if st.button("Reset"):
    st.session_state.clear()
    st.rerun()
