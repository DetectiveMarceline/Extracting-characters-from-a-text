import spacy
import nltk
from nltk import ne_chunk

# load both spacy and nltk

nlp_spacy = spacy.load("en_core_web_sm")
# download punkt
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# Tokenize and apply NER using spacy
text = """In the town of Digne, in the little village of Montreuil-sur-mer, there were, in 1815, two men who stood out, even in that city, which was famous for men of great stature. One of them was the mayor, M. Madeleine; the other, the Bishop of Digne. Let us say a word or two about them both. We will begin with the mayor. M. Charles-François-Bienvenu Myriel had been Bishop of Digne for only a short time when, in 1806, he went to visit the hospitals of his diocese. At that time the hospitals were nothing but mere infirmaries, so it was a day of great need.
 The curé, who was a good old man, told his wife about it: 'I saw a man today who is almost a saint."""
doc_spacy = nlp_spacy(text)
entities_spacy = [(ent.text, ent.label) for ent in doc_spacy.ents]

# Tokenize text for nltk
sentences = nltk.sent_tokenize(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

# apply NER using NLTK
entities_nltk = []
for sentence in tokenized_sentences:
    entities_nltk.extend(ne_chunk(nltk.pos_tag(sentence)))

# extract recognized entities from nltk
entities_nltk = [x for x in entities_nltk if isinstance(x, tuple)]
entities_nltk = [(entity[0], entity[1]) for entity in entities_nltk]

# combine

combined_entities = entities_nltk + entities_spacy

# for entity, lable in combined_entities:
# print(f"Name: {entity}    Label: {lable} ")


chars = [ent.text for ent in doc_spacy.ents if ent.label_ == "PERSON"]
print(chars)
