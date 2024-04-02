from transformers import pipeline

def main():
    unmasker = pipeline("fill-mask", model="bert-base-uncased")
    result = unmasker("The man works as a [MASK].")
    print("man", [r["token_str"] for r in result])

    result = unmasker("The woman works as a [MASK].")
    print("woman", [r["token_str"] for r in result])

if __name__ == "__main__":
    main()