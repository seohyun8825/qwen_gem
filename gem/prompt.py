import spacy

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def split_action_prompt(text: str):
    doc = _nlp()(text)
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    nouns = [token.lemma_ for token in doc if token.pos_ in ("NOUN", "PROPN")]

    verb = verbs[0] if verbs else "doing"
    obj = nouns[0] if nouns else "something"

    verb_prompt = f"A photo of a person {verb} something."
    object_prompt = f"A photo of a person using {obj}."
    action_prompt = f"A photo of a person {text.strip().rstrip('.')}"
    return verb_prompt, object_prompt, action_prompt
