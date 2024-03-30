from transformers import pipeline


def main():
    input = "In this course, we will teach you how to"

    # very small model
    generator = pipeline("text-generation", model="distilbert/distilgpt2")
    print("distilgpt2", generator(input, num_return_sequences=2, max_length=30))

    # very large model
    generator = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")
    print("mistral", generator(input, max_length=30))


if __name__ == "__main__":
    main()
