from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import time

tokenizer = AutoTokenizer.from_pretrained("ainize/klue-bert-base-mrc")
model = AutoModelForQuestionAnswering.from_pretrained("ainize/klue-bert-base-mrc")
# tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")
# model = AutoModelForQuestionAnswering.from_pretrained("monologg/koelectra-small-v2-distilled-korquad-384")

def model_inference(context : str, question : str):

    start = time.time()
    encodings = tokenizer(context, question, max_length=512, truncation=True,
                        padding="max_length", return_token_type_ids=False)
    encodings = {key: torch.tensor([val]) for key, val in encodings.items()}             

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    pred = model(input_ids, attention_mask=attention_mask)

    start_logits, end_logits = pred.start_logits, pred.end_logits

    token_start_index, token_end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)

    pred_ids = input_ids[0][token_start_index: token_end_index + 1]

    prediction = tokenizer.decode(pred_ids)
    end = time.time()

    print(f"모델 추론 시간 : {round(end-start,2)}s")

    return prediction
