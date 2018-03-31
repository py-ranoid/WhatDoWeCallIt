# What is this ?
Ideally an application that accepts a description about a product and generates a name.
Inspired by a very common question that arises while formulating an idea :  "What do we call it ?"

# How is it done ?
- Create an LSTM-based text generator
- Use the text generator to join 2 words
- Parse a (description) document to extract keywords
- Use Word2vec to find synonyms of Keywords
- Join these synonyms to generate potential names