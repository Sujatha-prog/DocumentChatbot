from flask import Flask, render_template, request
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

app = Flask(__name__)

model_name = "distilbert-base-cased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        user_input = request.form['user_input']
        document =  """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. It involves the development of algorithms and models that enable computers to understand, interpret, and generate human-like text.

One of the key tasks in NLP is Question Answering (QA), where a computer system is trained to answer questions posed by users. The distilbert-base-cased-distilled-squad model is a pre-trained model specifically designed for the Question Answering task.

To use this chatbot, you can ask it questions related to the provided document. For example:
- "What is the focus of Natural Language Processing?"
- "Which model is designed for Question Answering?"

Feel free to experiment with different questions and explore the capabilities of the chatbot.
""" # Replace with your document

        inputs = tokenizer(user_input, document, return_tensors="pt")
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True)