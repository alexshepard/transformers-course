from transformers import pipeline


def main():
    classifier = pipeline("zero-shot-classification")
    input = "This is a course about the Transfomers library"
    candidate_labels = ["education", "politics", "business"]
    print(classifier(input, candidate_labels))

    input = "That's great, it starts with an earthquake / Birds and snakes, and aeroplanes / And Lenny Bruce is not afraid"
    candidate_labels = ["education", "politics", "business", "lyrics"]
    print(classifier(input, candidate_labels))


if __name__ == "__main__":
    main()
