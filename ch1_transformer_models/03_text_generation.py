from transformers import pipeline


def main():
    generator = pipeline("text-generation")
    input = "In this course, we will teach you how to"
    print(generator(input))

    print(generator(input, num_return_sequences=2, max_length=15, truncation=True))


if __name__ == "__main__":
    main()
