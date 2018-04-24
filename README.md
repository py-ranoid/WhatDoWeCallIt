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
Example :

        Word 1 : brad
        Word 2 : angelina
        Bradlina      0.701200
        Brangelina    0.578985
        Angebrad      0.567822
        Braina        0.555771
        Bradelina     0.540182
        Bradgelina    0.511482
        Bradina       0.508800
        Angelrad      0.456382

# Startup Name generator testing
Requirements : Same as above + requests

Run `test_start_gen.py`.<br>
Enter a startup/product description.
Example :

        Enter a description of your product : A website to rent cameras
        ****
        0. rent + camera
        1. website + rent
        2. rent + website
        Which patterns would you like to drop?
        (Numbers seperated by spaces, Enter if all patterns are relevant)
        :1 2
        *******
        Word : camera
        0. shot         1. cut          2. picture      3. angle        4. scene        
        5. lens         6. control      7. move         8. camera       
        Which synonyms would you like to drop?
        (Numbers seperated by spaces, Enter if all words are relevant)
         :0 1 4 6 7        
        Enter a synonym for "camera" (Enter to pass) :
        *******
        Word : rent
        0. manor        1. lease        2. issue        3. assessment   4. income       
        5. mortgage     6. land         7. owner        8. rent         
        Which synonyms would you like to drop?
        (Numbers seperated by spaces, Enter if all words are relevant)
         :0 2 3 4 5 6 7
        Enter a synonym for "rent" (Enter to pass) :lend
        Enter a synonym for "rent" (Enter to pass) :borrow
        Enter a synonym for "rent" (Enter to pass) :

        Printing upto 15 names :
        Rentamera     0.674700
        Borramera     0.628271
        Borrowlens    0.623722
        Lendamera     0.610022
        Borrera       0.607200
        Lendlens      0.600271
        Borlens       0.597700
        Borrongle     0.586300
        Borrangle     0.579822
        Rentlens      0.579600
        Leamera       0.569271
        Rentcamera    0.553322
        Leascamera    0.533400
        Lendera       0.526571
        Leaseangle    0.518922





*Note : This model has been trained on popular US baby names and hence joins may reflect the same.*
