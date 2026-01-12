import torch
import shap
import streamlit as st
import numpy as np


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

HF_USERNAME = "priyanshis9876"  

TRANSLATOR_MODEL_ID = f"{HF_USERNAME}/translator"
ENG_SENTIMENT_MODEL_ID = f"{HF_USERNAME}/eng_sentiment"
SING_SENTIMENT_MODEL_ID = f"{HF_USERNAME}/sing_sentiment_singbert"

SINGLISH_EXAMPLES = [
    "Aiyo… whole day like that, really sian until cannot.",
    "Okay lor, like that we just settle here can already.",
    "I scared until my hand shaking sia.",
    "later boss see my mistake sure die one",
    "Ok lor, like that can already lah. Anything also can",
    "Eat here or makan there also same la, I okay one.",
    "Eh? Serious ah? I really cannot believe sia.",
    "Today mood damn off… really no strength to face anyone lor.",
    "Sian… every time like this, slowly I also no hope already",
    "Wah lao, you all never listen again, waste my time sia.",
    "Tonight must walk home alone, very scared one you know",
    "this place so dark, later got something jump out how",
    "Alamak this place so dark, later got something jump out how?",
    "Okay lah, like that settle lor. No big deal one.",
    "Wah piang eh, I never expect you to show up sia",
    "You suddenly shout like that I almost drop my heart",
    "Haiz feeling very sad... my dog passed away",
    "Wah today really damn shiok leh!",
    
]


import streamlit.components.v1 as components


def st_shap(plot_html: str, height: int | None = None):
    """Render a SHAP HTML string in Streamlit with styled background."""
    styled_html = f"""
    <head>{shap.getjs()}</head>
    <body>
        <div style="
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            {plot_html}
        </div>
    </body>
    """
    components.html(styled_html, height=height)


class SinglishReasoningExplainer:
    def __init__(self, model_id: str):
        """
        Initializes the model, tokenizer, pipeline and SHAP explainer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        self.emotion_mapping = {
            "LABEL_0": "anger",
            "LABEL_1": "fear",
            "LABEL_2": "joy",
            "LABEL_3": "neutral",
            "LABEL_4": "sadness",
            "LABEL_5": "surprise"
        }

        self.pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1,
        )

        self.explainer = shap.Explainer(self.pipe)

    def analyze_text(self, text: str):
        return self.explainer([text])

    def predict_top_label(self, text: str):
        preds = self.pipe(text)[0]
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)
        top = preds[0]
        label = top["label"]
        prob = top["score"]
        human_label = self.emotion_mapping.get(label, label)
        return label, human_label, prob, preds

    def generate_narrative(self, text: str, shap_values):
        top_label, top_emotion, top_conf, _ = self.predict_top_label(text)

        token_values = shap_values[0].values[:, self.label2id[top_label]]
        token_names = shap_values[0].data

        word_impacts = []
        for word, score in zip(token_names, token_values):
            if word.strip() == "":
                continue
            word_impacts.append((word.strip(), score))

        if not word_impacts:
            return "Could not build explanation for this input."

        word_impacts.sort(key=lambda x: x[1], reverse=True)

        drivers = [w for w in word_impacts if w[1] > 0]
        resistors = [w for w in word_impacts if w[1] < 0]

        narrative = f"""
**Input**: `{text}`  
**Prediction**: **{top_emotion.upper()}** (confidence {top_conf:.1%})

#### Reasoning Summary

The model reads this as **{top_emotion}**. Words with strong positive impact on this emotion:

"""

        for w, s in drivers[:3]:
            narrative += f"- `{w}` with impact **+{s:.3f}**\n"

        if resistors:
            worst_resistor = min(resistors, key=lambda x: x[1])
            narrative += (
                "\n**Words pulling in the opposite direction:**\n"
            )
            narrative += (
                f"- `{worst_resistor[0]}` with impact **{worst_resistor[1]:.3f}**\n"
            )
        else:
            narrative += (
                "\n✅ **No strong opposing words.** The sentiment is quite consistent.\n"
            )

        narrative += (
            "\n💡 *Singlish-specific tokens can carry emotional weight that might"
            " be lost when translated to plain English.*"
        )

        return narrative

    def shap_plot_with_labels(self, shap_values):
        """Generate SHAP plot with human-readable emotion labels"""
        if hasattr(shap_values, 'output_names') and shap_values.output_names is not None:
            original_names = shap_values.output_names
            shap_values.output_names = [
                self.emotion_mapping.get(name, name) 
                for name in original_names
            ]
        
        return shap.plots.text(shap_values, display=False)


@st.cache_resource
def load_components():
    translator = pipeline(
        "translation",
        model=TRANSLATOR_MODEL_ID,
        tokenizer=TRANSLATOR_MODEL_ID,
    )

    eng_sentiment_pipe = pipeline(
        "text-classification",
        model=ENG_SENTIMENT_MODEL_ID,
        tokenizer=ENG_SENTIMENT_MODEL_ID,
        return_all_scores=True,
    )

    singlish_explainer = SinglishReasoningExplainer(SING_SENTIMENT_MODEL_ID)

    return translator, eng_sentiment_pipe, singlish_explainer


def main():
    st.set_page_config(
        page_title="Singlish vs English Sentiment Demo", 
        layout="wide",
        page_icon="🇸🇬"
    )
    
    st.title("🇸🇬 Singlish vs English Sentiment Analysis")
    
    st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
    }
    .stSelectbox {
        margin-bottom: 20px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        Compare how sentiment analysis differs between two approaches:
        
        **Path 1**: Translate Singlish → English → Run English sentiment model  
        **Path 2**: Run Singlish sentiment model directly with explainability
        
        *Translation can lose crucial sentiment nuances!*
        """
    )

    selected_text = st.selectbox(
        "Select a Singlish sentence to analyze:",
        options=SINGLISH_EXAMPLES,
        index=0
    )
    user_input = st.text_input(
        "Or type your own Singlish sentence:",
        value="",
        placeholder="Type a Singlish sentence here (optional)..."
    )

    st.markdown("---")

    if st.button("Run Analysis", type="primary"):
        text_to_analyze = user_input.strip() if user_input.strip() else selected_text

        if not text_to_analyze:
            st.warning("Please select or type a sentence to analyze.")
        else:
            translator, eng_pipe, sing_explainer = load_components()

            with st.spinner("🔄 Analyzing sentiment across both models..."):
                translated = translator(text_to_analyze)[0]["translation_text"]

                eng_scores = eng_pipe(translated)[0]
                eng_scores = sorted(eng_scores, key=lambda x: x["score"], reverse=True)
                eng_top = eng_scores[0]

                sing_shap_values = sing_explainer.analyze_text(text_to_analyze)
                sing_label_id, sing_label, sing_conf, sing_all = sing_explainer.predict_top_label(
                    text_to_analyze
                )
                sing_narrative = sing_explainer.generate_narrative(
                    text_to_analyze, sing_shap_values
                )

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.header("🔄 Path 1: Translation → English Model")
            
            st.subheader(" Translated Text")
            st.info(translated)

            english_emotion_mapping = {
                "LABEL_0": "joy",
                "LABEL_1": "anger",
                "LABEL_2": "sadness",
                "LABEL_3": "fear",
                "LABEL_4": "surprise",
                "LABEL_5": "neutral",
            }

            eng_top_human = english_emotion_mapping.get(eng_top["label"], eng_top["label"])

            st.subheader("English Sentiment Prediction")
            
            st.metric(
                label="Detected Emotion",
                value=eng_top_human.upper(),
                delta=f"{eng_top['score']:.1%} confidence"
            )

            st.write("**Full Distribution:**")
            for item in eng_scores:
                human = english_emotion_mapping.get(item["label"], item["label"])
                percentage = item['score'] * 100
                st.progress(item['score'], text=f"{human.capitalize()}: {percentage:.1f}%")


        with col2:
            st.header("🇸🇬 Path 2: Direct Singlish Model")
            
            st.subheader(" Singlish Sentiment Prediction")
            
            st.metric(
                label="Detected Emotion",
                value=sing_label.upper(),
                delta=f"{sing_conf:.1%} confidence"
            )

            st.subheader(" Reasoning Narrative")
            st.markdown(sing_narrative)

            st.subheader("Token-Level Impact (SHAP)")
            try:
                plot = sing_explainer.shap_plot_with_labels(sing_shap_values)
                st_shap(plot, height=400)
            except Exception as e:
                st.warning(f"⚠️ Could not render SHAP text plot: {e}")
        
        st.markdown("---")
        st.subheader("📊 Key Differences")
        
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        
        with comp_col1:
            st.metric("Original Text Language", "Singlish")
        
        with comp_col2:
            eng_top_human = english_emotion_mapping.get(eng_top["label"], eng_top["label"])
            st.metric("English Model", eng_top_human.upper())
        
        with comp_col3:
            st.metric("Singlish Model", sing_label.upper())
        
        if eng_top_human.lower() != sing_label.lower():
            st.warning("⚠️ **Different predictions!** Translation may have altered the sentiment.")
        else:
            st.success("✅ Both models agree on the sentiment.")


if __name__ == "__main__":
    main()