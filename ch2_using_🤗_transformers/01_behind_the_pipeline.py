import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


def main():

    # preprocessing inputs with a tokenizer
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)

    # going through the model
    model = AutoModel.from_pretrained(checkpoint)

    # headless model outputs are high dimensional
    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

    # add a sequence classification head
    # now we get logits that are low dimensional (per the task)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)
    print(outputs.logits.shape)

    # postprocessing the output
    print(outputs.logits)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)
    print(model.config.id2label)
    for i in range(len(raw_inputs)):
        print(raw_inputs[i])
        print(
            list(zip(model.config.id2label.values(), predictions[i].detach().numpy()))
        )

    print()

    # try it with my own content
    raw_inputs = [
        "I listen to KEXP every morning, it helps me relax.",
        "I am confident that I can complete this NLP course.",
        "I am uncertain about the future.",
        "I am uncertain about the future, but I will try to be resiliant.",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    for i in range(len(raw_inputs)):
        print(raw_inputs[i])
        print(
            list(zip(model.config.id2label.values(), predictions[i].detach().numpy()))
        )


if __name__ == "__main__":
    main()
