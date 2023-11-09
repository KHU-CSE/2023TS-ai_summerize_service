from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta')
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')


def sentences_to_embeddings(sentences : list):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    embeddings, _ = model(**inputs, return_dict=False)
    return embeddings


