from transformers import pipeline

def main():
    ner = pipeline("ner", grouped_entities=True)
    input = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    print(input, ner(input))

    input = "That's great, it starts with an earthquake / Birds and snakes, and aeroplanes / And Lenny Bruce is not afraid"
    print(input, ner(input))

    ner = pipeline("ner", "vishnun/knowledge-graph-nlp", grouped_entities=True)
    input = "Alex is wondering what the point of all of this is."
    print(input, ner(input))

if __name__ == "__main__":
    main()