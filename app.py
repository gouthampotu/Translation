import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="Language Translator",
    page_icon="🌍",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-fr"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

# ----------------------------
# Language Codes
# ----------------------------

language_codes = {
    "English": "eng_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym"
}

# ----------------------------
# Translation Function
# ----------------------------

def translate(text, source_lang, target_lang):

    tokenizer.src_lang = language_codes[source_lang]

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
            language_codes[target_lang]
        ),
        max_length=256
    )

    output = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]

    return output


# ----------------------------
# UI
# ----------------------------

st.title("🌍 Language Translator")

st.write("Translate text using Hugging Face NLLB Model")

text = st.text_area(
    "Enter Text",
    height=150
)

col1, col2 = st.columns(2)

with col1:
    source = st.selectbox(
        "Source Language",
        list(language_codes.keys())
    )

with col2:
    target = st.selectbox(
        "Target Language",
        list(language_codes.keys()),
        index=1
    )

if st.button("Translate"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        with st.spinner("Translating..."):

            result = translate(
                text,
                source,
                target
            )

        st.success("Translation Completed!")

        st.subheader("Translated Text")

        st.text_area(
            "",
            value=result,
            height=150
        )
