import os
import time
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import pandas as pd
from datasets import load_dataset
import evaluate
from peft import PeftModel, PeftConfig

# Replace with your Hugging Face token
token = "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"
os.environ["hf_token"] = token

# Load summarization models
@st.cache_resource
def load_summarization_model(model_name, adapter_name=None):
    try:
        if model_name in [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct-Engineered"
        ]:
            print(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_auth_token=token, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            return tokenizer, model
        elif model_name in ["brianchu26/lora_llama3_finetuned_few_shot", "brianchu26/llama3_instruct_finetuned"]:
            base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # The base model used for fine-tuning
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_auth_token=token)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                use_auth_token=token, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            # Load LoRA adapters
            if adapter_name:
                model = PeftModel.from_pretrained(model, adapter_name, torch_dtype=torch.bfloat16, device_map="auto")
            return tokenizer, model
        elif model_name == "microsoft/Phi-3-mini-128k-instruct":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True
            )
            return tokenizer, model
        else:
            summarizer = pipeline("summarization", model=model_name, framework="pt", use_auth_token=token)
            return summarizer
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        print(f"Error loading model {model_name}: {e}")
        return None, None

# Load NLI model for hallucination detection
@st.cache_resource
def load_nli_model():
    model_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=token)
    nli_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, use_auth_token=token)
    return nli_pipeline

nli_model = load_nli_model()

df = pd.read_csv('train_with_titles.csv')
text1 = df.loc[0, 'dialogue']
summary1 = df.loc[0, 'soap']
text2 = df.loc[1, 'dialogue']
summary2 = df.loc[1, 'soap']

# Format messages for prompt-engineered Llama-3-8B
def format_messages_engineered(dialogue, data):
    dialogue_str = str(dialogue)
    messages = [
        {"role": "system", "content": "You are a doctor's assistant specializing in summarizing doctor-patient dialogues into detailed and accurate consultation notes. Your task is to provide a concise summary of the following dialogue and a corresponding treatment plan using the SOAP format (Subjective, Objective, Assessment, and Plan). Ensure you match the medical terms and follow the format of the examples provided."},
        {"role": "user", "content": text1},
        {"role": "assistant", "content": summary1},
        {"role": "user", "content": text2},
        {"role": "assistant", "content": summary2},
        {"role": "user", "content": dialogue_str},
    ]
    return messages

# Prepare prompt for the model
def prepare_prompt(dialogue, model_name, data=None):
    if model_name in [
        "meta-llama/Meta-Llama-3-8B-Instruct"   
    ]:
        dialogue_str = "\n".join(dialogue.split("\n"))
        messages = [
            {"role": "system", "content": "You are a doctor's assistant specializing in summarizing doctor-patient dialogues into detailed and accurate consultation notes. Your task is to provide a concise summary of the following dialogue and a corresponding treatment plan using the SOAP format (Subjective, Objective, Assessment, and Plan)."},
            {"role": "user", "content": "Summarize the following: " + dialogue_str},
        ]
        return messages
    elif model_name in ["meta-llama/Meta-Llama-3-8B-Instruct-Engineered", "brianchu26/lora_llama3_finetuned_few_shot", "brianchu26/llama3_instruct_finetuned"]:
        return format_messages_engineered(dialogue, data)
    elif model_name == "microsoft/Phi-3-mini-128k-instruct":
        dialogue_str = "\n".join(dialogue.split("\n"))
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": dialogue_str}
        ]
        return messages
    else:
        return dialogue

# Generate summary with Llama-3 with different decoding strategies
def generate_llama3_output(messages, tokenizer, model, temperature, top_p, top_k, max_new_tokens=512, strategy="Nucleus Sampling"): 
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("")
    ]

    if strategy == "Greedy":
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators[0]
        )
    elif strategy == "Beam Search":
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators[0],
            num_beams=5,  # You can adjust the number of beams
        )
    elif strategy == "Nucleus Sampling":
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators[0],
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    elif strategy == "Top-k Sampling":
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators[0],
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
        )

    response = outputs[0][input_ids.shape[-1]:]
    generated_summary = tokenizer.decode(response, skip_special_tokens=True)
    return generated_summary


# Generate summary with Phi-3
def generate_phi3_output(messages, tokenizer, model, temperature, top_p, top_k, max_new_tokens=500):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "return_full_text": False,
        "temperature": temperature,
        "do_sample": True,
        "top_p": top_p,
        "top_k": top_k,
    }
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

# Summarize dialogue
def summarize_text(dialogue, model_name, temperature, top_p, top_k, max_new_tokens, strategy, data=None):
    if model_name in [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct-Engineered",
        "brianchu26/lora_llama3_finetuned_few_shot", 
    ]:
        tokenizer, model = load_summarization_model(model_name, adapter_name="brianchu26/lora_llama3_finetuned_few_shot")  # Add the adapter path here
        if tokenizer and model:  # Ensure both are loaded correctly
            prompt = prepare_prompt(dialogue, model_name, data)
            return generate_llama3_output(prompt, tokenizer, model, temperature, top_p, top_k, max_new_tokens, strategy)
        else:
            st.error(f"Failed to load model: {model_name}")
            return None
    elif model_name == "brianchu26/llama3_instruct_finetuned": 
        tokenizer, model = load_summarization_model(model_name, adapter_name="brianchu26/llama3_instruct_finetuned")  # Add the adapter path here
        if tokenizer and model:  # Ensure both are loaded correctly
            prompt = prepare_prompt(dialogue, model_name, data)
            return generate_llama3_output(prompt, tokenizer, model, temperature, top_p, top_k, max_new_tokens, strategy)
        else:
            st.error(f"Failed to load model: {model_name}")
            return None
    elif model_name == "microsoft/Phi-3-mini-128k-instruct":
        tokenizer, model = load_summarization_model(model_name)
        if tokenizer and model:  # Ensure both are loaded correctly
            prompt = prepare_prompt(dialogue, model_name)
            return generate_phi3_output(prompt, tokenizer, model, temperature, top_p, top_k, max_new_tokens)
        else:
            return None
    else:
        summarizer = load_summarization_model(model_name)
        if summarizer:
            try:
                summary = summarizer(dialogue, max_length=100, min_length=50, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
                return summary[0]['summary_text']
            except Exception as e:
                st.error(f"Error generating summary: {e}")
                return None
        else:
            return None

def check_hallucination_nli(dialogue, summary):
    input_text = f"premise: {dialogue}\nhypothesis: {summary}"
    try:
        result = nli_model(input_text)[0]  # Assuming the pipeline returns a list of dictionaries
        
        # Extract probabilities directly using dictionary comprehension
        scores = {res['label'].upper(): res['score'] for res in result}
        contradiction_prob = scores.get('CONTRADICTION', 0)
        entailment_prob = scores.get('ENTAILMENT', 0)
        neutral_prob = scores.get('NEUTRAL', 0)

        st.write("### Hallucination Detection Results")
        st.write(f"**Contradiction Probability:** {contradiction_prob:.2f}")
        st.write(f"**Entailment Probability:** {entailment_prob:.2f}")
        st.write(f"**Neutral Probability:** {neutral_prob:.2f}")

        # Consider a summary as hallucinated if the contradiction probability is the highest
        is_hallucinated = contradiction_prob > max(entailment_prob, neutral_prob)
        return is_hallucinated

    except Exception as e:
        return False


def evaluate_summaries(reference_summaries, generated_summaries):
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    bertscore = evaluate.load('bertscore')

    rouge_result = rouge.compute(predictions=generated_summaries, references=reference_summaries)
    meteor_result = meteor.compute(predictions=generated_summaries, references=reference_summaries)
    bertscore_result = bertscore.compute(predictions=generated_summaries, references=reference_summaries, lang="en")

    return {
        "ROUGE": rouge_result,
        "METEOR": meteor_result,
        "BERTScore": bertscore_result
    }

def format_summary(summary):
    # Replace 'S:', 'O:', 'A:', 'P:' with new lines followed by the respective labels
    summary = summary.replace("S:", "\n\n**Subjective**\n")
    summary = summary.replace("O:", "\n\n**Objective**\n")
    summary = summary.replace("A:", "\n\n**Assessment**\n")
    summary = summary.replace("P:", "\n\n**Plan**\n")
    return summary

# Generate summary and check for hallucination
def generate_summary(dialogue, selected_model, temperature, top_p, top_k, max_new_tokens, strategy, data=None):
    summary = summarize_text(dialogue, selected_model, temperature, top_p, top_k, max_new_tokens, strategy, data)
    if summary:
        summary = format_summary(summary)
        st.write("### Generated Clinical Note")
        st.markdown(
            f"""
            <div style='background-color: #f0f0f5; padding: 10px; border-radius: 10px; margin-bottom: 20px;'>
                <p style='font-family:Arial; font-size:16px;'>{summary}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.write(" ")
        st.write(" ")
        
        with st.spinner("Checking for hallucinations..."):
            time.sleep(2)
            is_hallucinated = check_hallucination_nli(dialogue, summary)
            
            st.write(" ")
            st.write(" ")
            
            while is_hallucinated:
                st.error("The summary is hallucinated.")
                st.write("Regenerating summary in 3 seconds...")
                time.sleep(3)
                generate_summary(dialogue, selected_model, temperature, top_p, top_k, max_new_tokens, data)
            else:
                st.success("The summary is good.")
                
        reference_summaries = [dialogue]
        generated_summaries = [summary]
        scores = evaluate_summaries(reference_summaries, generated_summaries)
        
        st.write("### Evaluation Scores")
        st.write(f"**ROUGE Scores:**")
        for key, value in scores['ROUGE'].items():
            if isinstance(value, list):
                value = value[0]
            if isinstance(value, (int, float)):
                st.write(f"{key}: {value:.2f}")
            else:
                st.write(f"{key}: {value}")

        st.write(f"**METEOR Score:** {scores['METEOR']['meteor']:.2f}")
        
        st.write(f"**BERTScore:**")
        for key, value in scores['BERTScore'].items():
            if isinstance(value, list):
                value = value[0]
            if isinstance(value, (int, float)):
                st.write(f"{key}: {value:.2f}")
            else:
                st.write(f"{key}: {value}")
    else:
        st.error("Failed to generate summary.")

st.title("Doctor-Patient Dialogue to Clinical Note Generator🤒")

# Load example dialogues
# Now using the 'Title' column as keys
example_dialogues = {df.loc[i, 'Title']: df.loc[i, 'dialogue'] for i in range(len(df))}
example_titles = list(example_dialogues.keys())

st.header("Select Example Dialogue or Input a Dialogue Transcript of Your Own")
example_choice = st.selectbox("Choose an example dialogue:", [""] + example_titles)
if example_choice:
    user_dialogue = example_dialogues[example_choice]
else:
    user_dialogue = ""

st.header("Input Dialogue")
user_dialogue = st.text_area("Enter the doctor-patient dialogue here:", value=user_dialogue, height=300)

st.header("Model Selection")
model_choice = st.selectbox(
    "Select the summarization model:",
    ["Llama-3-8B-Instruct", "Llama-3-8B-Engineered", "Phi-3", "Fine-Tuned BART", "Fine-Tuned Llama3", "Fine-Tuned Llama3-Instruct"]
)

model_mapping = {
    "Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Llama-3-8B-Engineered": "meta-llama/Meta-Llama-3-8B-Instruct", 
    "Phi-3": "microsoft/Phi-3-mini-128k-instruct",
    "Fine-Tuned BART": "knkarthick/MEETING_SUMMARY",
    "Fine-Tuned Llama3": "brianchu26/lora_llama3_finetuned_few_shot",
    "Fine-Tuned Llama3-Instruct": "brianchu26/llama3_instruct_finetuned"
}

st.header("Generation Parameters")
temperature = st.slider("Temperature:", 0.0, 1.0, 0.7)
top_p = st.slider("Top-p (nucleus sampling):", 0.0, 1.0, 0.9)
top_k = st.slider("Top-k:", 0, 100, 50)

# Add decoding strategy selection
st.header("Decoding Strategy")
decoding_strategy = st.selectbox(
    "Select the decoding strategy:",
    ["Greedy", "Beam Search", "Nucleus Sampling", "Top-k Sampling"]
)

# Button to generate summary
if st.button("Generate Summary"):
    if user_dialogue:
        with st.spinner("Generating summary..."):
            selected_model = model_mapping[model_choice]
            generate_summary(user_dialogue, selected_model, temperature, top_p, top_k, max_new_tokens=512, strategy=decoding_strategy)
    else:
        st.warning("Please enter a doctor-patient dialogue.")

if st.button("Reset"):
    st.session_state.clear()
    st.rerun()
