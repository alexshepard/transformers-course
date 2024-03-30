from transformers import pipeline


def main():
    question_answerer = pipeline("question-answering")
    question = "Where do I work?"
    context = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    print(question, context, question_answerer(question=question, context=context))

    context = "I make sketches for a variety of entertainment organizations."
    print(question, context, question_answerer(question=question, context=context))

    context = "I am self employed."
    print(question, context, question_answerer(question=question, context=context))

    question = "How large are northern alligator lizards?"
    # thanks wikipedia
    context = """
        The northern alligator lizard (Elgaria coerulea) is a species of medium-sized lizard 
        in the family Anguidae. The species is endemic to the North American west coast.
        
        The southern alligator lizard (Elgaria multicarinata) is a common species of lizard 
        in the family Anguidae. The species is native to the Pacific coast of North America.
        It ranges from Baja California to the state of Washington and lives in a variety of 
        habitats including grasslands, chaparral, forests, and even urban areas. In dry climates, 
        it is likely to be found in moist areas or near streams. There are five recognized 
        subspecies.
        """
    print(question, context, question_answerer(question=question, context=context))


if __name__ == "__main__":
    main()
