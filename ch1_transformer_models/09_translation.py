from transformers import pipeline


def main():
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    input = "Ce cours est produit par Hugging Face."
    output = translator(input)
    print(output)

    translator = pipeline("translation_en_to_fr")
    print(translator(output[0]["translation_text"]))


if __name__ == "__main__":
    main()
