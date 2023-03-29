import spacy


def wakati_proc(sentence_list):
    nlp = spacy.load('ja_ginza')
    list_aa = list(nlp.pipe(sentence_list))

    docs = []
    for unit_aa in list_aa:
        unit_bb = []
        for word in unit_aa:
            unit_bb.append(str(word))
        text = ' '.join(unit_bb)
        docs.append(text)
#   print(docs)
#
    return docs
