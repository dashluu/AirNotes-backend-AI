from transformers import pipeline

summary_model = "facebook/bart-large-cnn"
qa_model = "deepset/roberta-base-squad2"
summarizer = pipeline("summarization", model=summary_model)
qa = pipeline("question-answering", model=qa_model, tokenizer=qa_model)


def summarize(text):
    tokens = summarizer.tokenizer.encode(text)
    max_len = int(len(tokens) * 0.7)
    min_len = int(len(tokens) * 0.2)
    result = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return result[0]["summary_text"]


def answer_question(question, context):
    qa_input = {
        "question": question,
        "context": context
    }
    result = qa(qa_input)
    return result
