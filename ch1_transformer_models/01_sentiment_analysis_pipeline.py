from transformers import pipeline

def main():
    classifier = pipeline("sentiment-analysis")
    input = "I've been waiting for a HuggingFace course my whole life."
    print(input, classifier(input))

    inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!"
    ]
    print(list(zip(inputs, classifier(inputs))))

if __name__ == "__main__":
    main()