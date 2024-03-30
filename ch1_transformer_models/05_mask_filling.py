from transformers import pipeline


def main():
    base_input = "This course will teach you all about {} models."

    unmasker = pipeline("fill-mask")
    mask_word = "<mask>"
    print(
        "distilbert/distilroberta-base", unmasker(base_input.format(mask_word), top_k=2)
    )

    unmasker = pipeline("fill-mask", "google-bert/bert-base-cased")
    mask_word = "[MASK]"
    print(
        "google-bert/bert-base-cased", unmasker(base_input.format(mask_word), top_k=2)
    )


if __name__ == "__main__":
    main()
