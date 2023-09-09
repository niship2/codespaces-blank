import spacy
import sentencepiece as spm
sp = spm.SentencePieceProcessor()

def sentencep(dataframe,col):
    # @title txtファイルをcsvに移動
    #df = df.assign(txt=df['txt'].str.split('【請求項')).explode('txt')
    dataframe[col].to_csv("temptxt.txt")
    spm.SentencePieceTrainer.Train("--input=temptxt.txt --model_prefix=trained_model --vocab_size=3000")
    sp.Load("trained_model.model")

    dataframe['txt_enc'] = dataframe[col].replace('d+|[０-９]+','###',regex=True).fillna('-').apply(sp.EncodeAsPieces).apply(" ".join)
    return dataframe



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
