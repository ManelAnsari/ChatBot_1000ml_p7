# Travel Bot - Discord Productionized
## Overview
The purpose of this project is to create a Chatbot for helping individuals book trips by plane. The Chatbot is able to hold small conversations, usually so long as the person is detailed in the context. When the user asks to book a flight, it will begin asking more pointed question to get information about the flight, including Destination, Origin, Return and Departure dates and the budget. Once it has the information, it will confirmed the information and offer for the user to change it. Once confirmed, the program looks that info up in the flight API (Amadeus) and returns 5 detail options for flights, sorted by price (ascending).

## Main Files
### traveler.py
This is the main script file used to run the bot when connected to Discord. 

### Travelbot2.ipynb
This is the present final notebook of the chatbot. The notebook contains all relevant packages, libraries and installs needed. It trains the chatbot (used from the Chatterbot library), works through some NER for information extraction and outputs a data pull from a travel API. It can be run to test the working of all the helper functions in the same notebook.

There are plenty areas for improvement, discussed in the corresponding section.
### Project7_Plan_Notes.xlsx
A project plan outlook for the first portion of the project (getting the bot working). There were many roadblocks along the way, and more time was used on actually developing the chatbot. 

The second page was a plan for possible intents. Ultimately that method was not pursued.
### Airports.csv
This file contained information translating cities into IATA codes that could be used in the Amadeus API for flight booking. This is a small list. For future works, consider looking into ways to pull this information from another API. This would require more functionality in the bot however, since there are often more than one airport in a city. The bot would then require more flexible ways of communicating for these pruposes. 

### Travel_Chatbot_Deck.pptx
This is my presentation deck for the short presentation I gave on the first portion of my project.

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
7. All of this was gathered and output to Discord to interact with the user. At each point in the script, there are keywords that will output a specific string and change the state of the bot. Those keywords are output throughout the bots usage by a user. 

## Next Steps
There are many next steps for this project. As it was my first attempt at anything NLP related, and the time frame was about 2 weeks, I think this was fairly well done, despite the shortcomings I list now.
1. Training data was not well utilized. There is certainly a better way to make it useful. Perhaps try something more along the lines of NLU, though this may require more computing power. This will also aid in having conversations appear more natural. Presently, if a user uses the bot more than once, they may begin to notice the rigidity of what it can say. 
3. Pattern matching was done well, though it could be more precise and more broadly applied. There must be a better way to assign context to different sentences of the user. 
4. Consider including an IATA code database for access to more flights. This would require a deeper level of coding, since the bot would then need to know which airport in a particular city a user wanted (if there were more than one). This ended up outside the scope of what could be accomplished.
5. Consider using a different library for the chatbot. Because Chatterbot was used, there are some package requirements that were necessarily outdated and thus present security risks.

## Discord Productionization
To productionize this bot, the Discord app was used. The API documentation for Discord is here: https://discordpy.readthedocs.io/en/latest/api.html. Furthermore, the basics on how to make a bot in python were learned from the following tutorial https://realpython.com/how-to-make-a-discord-bot-python/. 

## Summary and Conclusions
This Discord bot was produced by me, as a Data Scientist working at 1000ml. It functions at the capacity that I expected and worked towards over about a month, and does its job well enough. There are certainly many improvements that could be made, but as a first attempt at a chatbot, I think this will suffice. Feel free to clone/fork the repo if you would like to try it out for yourself. 
