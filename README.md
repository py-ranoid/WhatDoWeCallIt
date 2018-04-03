# What is this ?
Ideally an application that accepts a description about a product and generates a name.
Inspired by a very common question that arises while formulating an idea :  "What do we call it ?"

# How is it done ?
- Create an LSTM-based text generator (done)
- Use the text generator to join 2 words (done)
- Parse a (description) document to extract keywords
- Use Word2vec to find synonyms of Keywords
- Join these synonyms to generate potential names


# Bridge testing
Requirements

- numpy
- keras
- Tensorflow
- pandas

Run `test_bridge.py`.<br>
Enter 2 words (ideally names) you'd like to join.
Names **must** be 3 letters or greater.

*Note : This model has been trained on popular US baby names and hence joins may reflect the same.*
