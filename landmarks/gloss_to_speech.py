import os
import time
import re
import nltk
import json
from tts_module import create_synthesizer

# Ensure necessary NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# PURE NLP ENGINE - LINGUISTIC PATTERN RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

# Technical gloss expansion still needed for dataset labels
BASE_EXPANSION_MAP = {
    "i_me_mine_my": "I",
    "keepsmile": "keep smiling",
    "thank": "thank you",
    "good morning": "good morning",
    "happy birthday": "happy birthday",
    "how are you": "how are you",
    "i need help": "I need help"
}

# Manual POS overrides for ISL dataset labels that NLTK might mistag
BASE_POS_OVERRIDE = {
    "thirsty": "JJ", "sick": "JJ", "careful": "JJ",
    "again": "RB",
    "book": "NN", "file": "NN", "team": "NN", "attendance": "NN",
    "home": "NN", "seat": "NN",
    "where": "WRB", "how": "WRB", "what": "WP"
}

def _load_allowed_glosses():
    base = os.path.join(os.path.dirname(__file__), "models", "label_map.json")
    if not os.path.exists(base):
        return set()
    with open(base, "r") as f:
        data = json.load(f)
    return set(data.keys())

ALLOWED_GLOSSES = _load_allowed_glosses()
EXPANSION_MAP = {k: v for k, v in BASE_EXPANSION_MAP.items() if k in ALLOWED_GLOSSES}
POS_OVERRIDE = {k: v for k, v in BASE_POS_OVERRIDE.items() if k in ALLOWED_GLOSSES}

AUX_WORDS = {
    "is", "am", "are", "was", "were", "be", "being", "been",
    "do", "does", "did",
    "have", "has", "had",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must"
}

def pure_nlp_processor(gloss_string, models_dir="models"):
    """
    Pure NLP Engine: Uses NLTK POS tagging to perform linguistic reconstruction.
    Moves away from hardcoded sign rules to general grammar patterns.
    """
    if not gloss_string:
        return ""

    # 1. Expand Multi-word Glosses and Tokenize
    gloss_string = gloss_string.encode('ascii', 'ignore').decode('ascii').strip().lower()
    
    # Sort keys by length descending to match longest glosses first (e.g. "on the way" before "on")
    sorted_glosses = sorted(EXPANSION_MAP.keys(), key=len, reverse=True)
    
    expanded_temp = gloss_string
    for g in sorted_glosses:
        pattern = r'\b' + re.escape(g) + r'\b'
        expanded_temp = re.sub(pattern, EXPANSION_MAP[g], expanded_temp)
    
    expanded_words = re.split(r'\s+', expanded_temp)
    expanded_words = [w for w in expanded_words if w.strip()]
    
    expanded_words = [w for w in expanded_words if w.lower() != "break"]
    expanded_words = [w for w in expanded_words if w.lower() not in {"no", "sign", "performed"}]
    
    # 2. Automated Part-of-Speech Tagging
    tagged = nltk.pos_tag(expanded_words)
    
    # Apply manual overrides for dataset-specific accuracy
    corrected_tags = []
    for word, tag in tagged:
        lw = word.lower()
        corrected_tag = POS_OVERRIDE.get(lw, tag) if lw in ALLOWED_GLOSSES else tag
        corrected_tags.append((word, corrected_tag))
    
    print(f"  [POS Tags]: {corrected_tags}")

    # 3. Grammatical Pattern Reconstruction
    # Pattern Logic: 
    # [PRP] + [JJ] -> Insert 'am/is/are'
    # [PRP] + [VB] -> Insert 'want to'
    # [NN] + [WRB] -> Invert (Where is the NN?)
    
    processed = []
    is_query = any(t in ["WRB", "WP"] for w, t in corrected_tags)
    
    if is_query and corrected_tags:
        wh_idx = next((i for i, (_, t) in enumerate(corrected_tags) if t in ["WRB", "WP"]), -1)
        if wh_idx == 0 and len(corrected_tags) >= 2:
            wh_word = corrected_tags[0][0]
            rest = [w for (w, _) in corrected_tags[1:]]
            rest_phrase = " ".join(rest).strip()
            article_nouns = {"book", "file", "team", "attendance", "sun", "home", "school", "work", "office", "place", "seat"}
            if rest and corrected_tags[1][1].startswith("NN") and rest[0].lower() in article_nouns:
                rest_phrase = f"the {rest_phrase}"
            sentence = f"{wh_word} is {rest_phrase}".strip()
            sentence = sentence[0].upper() + sentence[1:] if sentence else ""
            if not sentence.endswith("?"):
                sentence += "?"
            return sentence
    
    i = 0
    while i < len(corrected_tags):
        word, tag = corrected_tags[i]
        
        # Article insertion for known Nouns
        # Nouns like 'book', 'file', 'team' often need 'the'
        if tag == "NN" and word.lower() in ["book", "file", "team", "attendance", "sun", "home", "school", "work", "office", "place", "seat"]:
            processed.append("the")
            
        processed.append(word)

        if i + 1 < len(corrected_tags):
            next_word, next_tag = corrected_tags[i+1]
            
            # Pattern: Pronoun + Adjective -> Copula (am/is/are)
            if tag == "PRP" and next_tag.startswith("JJ"):
                if word.lower() == "i": processed.append("am")
                elif word.lower() in ["you", "we", "they"]: processed.append("are")
                else: processed.append("is")
            
            # Pattern: Pronoun + Verb-ing (Present Continuous) -> am/is/are + VBG
            elif tag == "PRP" and next_tag == "VBG":
                if word.lower() == "i": processed.append("am")
                elif word.lower() in ["you", "we", "they"]: processed.append("are")
                else: processed.append("is")

            # Pattern: Pronoun + Verb (Base/Present) -> Modal smoothing (want to)
            elif tag == "PRP" and next_tag in ["VB", "VBP"]:
                # Avoid inserting 'want to' if the verb is already a copula or helper
                helpers = ["am", "is", "are", "was", "were", "be", "do", "does", "did", "have", "has", "had", "can", "will"]
                if word.lower() == "i" and next_word.lower() not in helpers:
                    processed.append("want to")
            
            # Pattern: Pronoun + Preposition/Location -> Insert 'am/is/are'
            elif tag == "PRP" and (next_tag == "IN" or next_word.lower() in ["home", "school", "work", "office", "place", "seat"]):
                if word.lower() == "i": processed.append("am")
                elif word.lower() in ["you", "we", "they"]: processed.append("are")
                else: processed.append("is")
                
                # Special case: if it's not a preposition yet but a known location, add 'at'
                if next_tag != "IN": processed.append("at")
            
            # Pattern: Noun + Adjective -> Copula (is/are)
            elif tag in ["NN", "NNS"] and next_tag.startswith("JJ"):
                if tag == "NNS": processed.append("are")
                else: processed.append("is")

        i += 1

    # 4. Synthesis & Finishing
    sentence = " ".join(processed)
    
    # Special handling for 'congratulations' to avoid 'I congratulations you'
    toks = [w.lower() for w, _ in corrected_tags]
    if "congratulations" in toks:
        if "you" in toks:
            sentence = "Congratulations to you"
        else:
            sentence = "Congratulations"
    
    # Handling ISL Query Order (e.g., "Book where" -> "Where is the book")
    if is_query:
        query_word = next((w for w, t in corrected_tags if t in ["WRB", "WP"]), "")
        if sentence.lower().endswith(query_word.lower()) and len(processed) > 1:
            # Shift query word to front
            core_sentence = sentence[:-(len(query_word))].strip()
            sentence = f"{query_word} is {core_sentence}"
        
        # Also handle WH word already at front with missing copula
        tokens = sentence.split()
        if tokens and tokens[0].lower() == query_word.lower():
            if len(tokens) >= 2 and tokens[1].lower() not in ["is", "are", "am", "was", "were"]:
                article_nouns = {"book", "file", "team", "attendance", "sun", "home", "school", "work", "office", "place", "seat"}
                if tokens[1].lower() in article_nouns:
                    sentence = f"{tokens[0]} is the " + " ".join(tokens[1:])
                else:
                    sentence = f"{tokens[0]} is " + " ".join(tokens[1:])
        
        if not sentence.endswith("?"):
            sentence = sentence.strip() + "?"
    
    # Final Polish
    sentence = sentence.strip().capitalize()
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
        
    # Clean up artifacts
    sentence = re.sub(r"\bI I\b", "I", sentence, flags=re.I)
    sentence = sentence.replace("i am", "I am").replace("i want", "I want")
    
    return sentence

def main():
    gloss_file = os.path.join(os.path.dirname(__file__), "recorded_predictions.txt")
    
    if not os.path.exists(gloss_file):
        print(f"Error: {gloss_file} not found. Please run recorded_predictions.py first.")
        return

    with open(gloss_file, "r") as f:
        gloss_content = f.read().strip()
    
    if not gloss_content:
        print("The gloss file is empty.")
        return

    print(f"\n--- Pure NLP Translation Engine (NLTK) ---")
    print(f"Input Glosses: {gloss_content}")
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    sentence = pure_nlp_processor(gloss_content, models_dir=models_dir)
    print(f"Output English: {sentence}")
    
    print("\n--- Synthesizing Speech ---")
    synth = create_synthesizer(rate=180)
    
    if synth:
        synth.speak(sentence)
        time.sleep(2.0 + (len(sentence) / 10))
        synth.close()
        print("Speech completed.")
    else:
        print("Warning: Could not initialize TTS engine.")

if __name__ == "__main__":
    main()
