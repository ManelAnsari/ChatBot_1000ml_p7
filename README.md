# Part 1
## Overview
The purpose of this project is to create a Chatbot for helping individuals book trips by train or plane. The Chatbot should
be able to have a regular conversation with the customer in English, and in doing so, ask for and extract the trip information
i.e. intended trip time, destination and point of origin, duration of trip and monetary budget.

## Main Files
### Travel_Chatbot.ipynb
This is the present final product of the chatbot. The notebook contains all relevant packages, libraries and installs needed. It trains the chatbot (used from the Chatterbot
library), works through some NER for information extraction and outputs a data pull from a travel API. 

There are plenty areas for improvement, discussed in the corresponding section.
### Project7_Plan_Notes.xlsx
A project plan outlook. There were many roadblocks along the way, and more time was used on actually developing the chatbot. 

The second page was a plan for possible intents. Ultimately that method was not pursued.
### Airports.csv
This file contained information translating cities into location codes that could be used in the Amadeus API for flight booking

### Travel_Chatbot_Deck.pptx
This is my presentation deck for the short presentation I gave on my project.

## Training Data
The data used for training is all contained within the notebook. Some sample conversations, greetings and english corpus data from the Chatterbot library. 

I attempted to use other training data but since the chatbot simply pulls out full sentences in response to a user, they were often not useful. More work is required to massage these things out.

Some example training data that is not included but attempted in training are the RSICS dataset and the Ubuntu Dialogue Corpus, both of which are easily accessible online, but were too large to be included in this repository.

## Methodology
The methodology is well laid out in the main notebook. I will do so again here:
1. Packages/Imports/Installs
2. Assemble training data. Created some myself, and used the English corpus from the Chatterbot Library.
3. Trained the bot using ListTrainer and CorpusTrainer as per the Chatterbot class. Lists were the conversations I provided, and the Corpus as mentioned above.
4. NER (Named Entity Recognition) performed with SpaCy pattern matching. This was done individually for the budget, depart/return dates and origin/destination locations. 
5. The values were pulled out of partially scripted conversation and input into the API caller (Amadeus).
6. The API call returned a flight booking and ended the program. 

## Next Steps
There are many next steps for this project. As it was my first attempt at anything NLP related, and the time frame was about 2 weeks, I think this was fairly well done, despite the shortcomings I list now.
1. Training data was not well utilized. There is certainly a better way to make it useful. Perhaps try something more along the lines of NLU, though this may require more computing power.
2. There are multiple instances of scripted responses. This isn't really the direction I wanted, though the issue was that the chatbot would simply return full sentences out of the training data, which was not ideal with the data that was being used previously.
3. Pattern matching was done well, though it could be more precise and more broadly applied. Precise in how it extracts the information, and broadly applied in that it doesn't need 5 sentences for 5 pieces of information. 
4. Finally, I would have prefered to implement a looping structure that allowed corrections to be made to the flight booking information should the user not be happy with it. 

# Part 2
On its way in the next few weeks!
## The GUI/Productionization

## Summary and Conclusions
