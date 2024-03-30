from transformers import pipeline

classifier = pipeline("sentiment-analysis")
input = "I've been waiting for a HuggingFace course my whole life."
print(classifier(input))
