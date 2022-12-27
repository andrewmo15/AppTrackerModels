import spacy

def getCompanyFromText(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    for entity in doc.ents:
        print(entity, entity.label_)
    return None