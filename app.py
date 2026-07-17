import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="Language Translator",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Language Translator")
st.write("Translate text using Facebook NLLB-200")

device = "cuda" if torch.cuda.is_available() else "cpu"

languages = {
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


@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()


def translate(text, source, target):

    tokenizer.src_lang = languages[source]

    encoded = tokenizer(text, return_tensors="pt")

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
            languages[target]
        ),
        max_length=200
    )

    translated = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return translated[0]


text = st.text_area("Enter Text", height=150)

col1, col2 = st.columns(2)

with col1:
    source = st.selectbox(
        "Source Language",
        list(languages.keys())
    )

with col2:
    target = st.selectbox(
        "Target Language",
        list(languages.keys()),
        index=5
    )

if st.button("Translate"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        with st.spinner("Translating..."):

            result = translate(text, source, target)

        st.success("Translation Completed")

        st.text_area(
            "Translated Text",
            result,
            height=150
        )
