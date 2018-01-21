import nltk


def tokenize_sentences(sentences, words_dict):
    tokenized_sentences = []
    for sentence in sentences:
        tokens = nltk.tokenize.word_tokenize(sentence.decode("utf-8"))
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict