from transformers import pipeline


def main():
    summarizer = pipeline("summarization")
    input = """
        America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science. As a result, there 
        are declining offerings in engineering subjects dealing with infrastructure, 
        the environment, and related issues, and greater concentration on high 
        technology subjects, largely supporting increasingly complex scientific 
        developments. While the latter is important, it should not be at the expense 
        of more traditional engineering.

        Rapidly developing economies such as China and India, as well as other 
        industrial countries in Europe and Asia, continue to encourage and advance 
        the teaching of engineering. Both China and India, respectively, graduate 
        six and eight times as many traditional engineers as does the United States. 
        Other industrial countries at minimum maintain their output, while America 
        suffers an increasingly serious decline in the number of engineering graduates 
        and a lack of well-educated engineers.
    """
    print(summarizer(input))

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
    # nope
    print(summarizer(context))

    # try again. still nope
    summarizer = pipeline("summarization", "facebook/bart-large-cnn")
    print(summarizer(context))

if __name__ == "__main__":
    main()
