path='data/'
#import colab_env
import os
import datetime
from dotenv import load_dotenv
load_dotenv(dotenv_path='')
# The API
from amadeus import Client, ResponseError

# This library can extract dates from text
import dateutil.parser as dparser

# We'll need the following nltk packages
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# We'll use lemmas instead of stems.
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# We will tag the parts of the speech 
from nltk.tag import pos_tag

# Get the stop words
from nltk.corpus import stopwords
nltk.download('stopwords')

# We'll need json to open a file of intents
import json

# We will need to save the model
import pickle

# Some helper libraries
import numpy as np
import pandas as pd
import random
import re

# We'll need these to actually create a machine learning model.
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adagrad, Adam, Adadelta

# We may need to try out a grid search
from sklearn.model_selection import GridSearchCV

# We'll need this for the in-code tagging of words
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import Matcher

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

# Greetings
hellos = [['Hello!','Hi there'],
          ['Hi there','Hello!'],
          ['Greetings','Hey'],
          ['Hey','Oh hey!'],
          ['Good morning','Good morning'],
          ['Good evening','Good evening'],
          ['Hey there!','Well hello there']
         ]

# Farewells
byes = [['Bye','Goodbye!'],
        ['See ya','Good bye!'],
        ['Have a good night','You as well, goodbye!'],
        ['Farewell','Later!'],
        ['Goodbye!', 'Bye for now!']
       ]

convo1=["I would like to book a flight.",
        "I can help you with that. Where are you traveling to?",
        "I am traveling to Singapore.",
        "What date will you be traveling?",
        "I want to fly on June 14th.",
        "When would you like to return?"]

convo2=['I want to buy a plane ticket.',
        'I can help you make your reservation. What is your destination?',
        'My final destination is Sydney, Australia.',
        'What is your travel date?',
        'I am making a reservation for December 12th.',
        'Fantastic, when will you need to come back?']

convo3=['I need to make a plane reservation.',
        'We can book your trip right now. What city are you flying to?',
        'I need to fly to New York City.',
        'What date would you like me to book this plane ticket(s) for?',
        'I need a flight on July 4th.',
        "Awesome, when do you think you'd like to return?"]

convo4=['I want to take a vacation',
        'We can help you with that! What were you thinking?',
        'I was thinking Dominica.',
        'When would you like to leave?',
        'I need a flight on March 3rd.',
        "Let's get booking then! I'll need your return date."]
convo5=['I need a goddamn vacation',
        'Where do you come from, where do you go? Where are you coming from, Cotton Eyed Joe?',
        "That was random, but I think I'll check out London",
        "Radical, let's get to booking! When did you want to leave?",
        "Probably this week, June 29th, 2020",
        "Excellent, I only have a few more questions for you."
        ]

convos = [convo1,convo2,convo3, convo4, convo5]

# We will first train the bot with the lists of data we've given it above.
chatbot = ChatBot('Traveller',logic_adapters=["chatterbot.logic.BestMatch"])
travel_listtrainer = ListTrainer(chatbot)

for convo in convos:
  travel_listtrainer.train(convo)

for hi in hellos:
  travel_listtrainer.train(hi)

for bye in byes:
  travel_listtrainer.train(bye)

# Next we will need to train in with the corpus.
travel_corpustrainer = ChatterBotCorpusTrainer(chatbot)

travel_corpustrainer.train("chatterbot.corpus.english")


# We'll note here that I'll have to consider what happens when the bot asks for the second date. Does it run through this again, or do I create a second function. 
# This will have to be changed depending on the global variables I choose to inlcude
def get_dates(message):
    '''
    This function checks if there are any dates in the message, if so it pulls them out. If there are two dates, then we need to
    parse the list more closely to pull out each date. This function assumes that the date(s) was/were written in a readable form.
    '''
    # Remove punctuation
    clean_message = re.sub(r'[^\w\s]','',message)
    # The first try checks if there are some dates. If there are none or two, it throws a ValueError
    # The second try assumes that there were actually two dates, if this is not true, then it returns nothing.
    try:
        depart_date = dparser.parse(clean_message,fuzzy=True)
        return_date = None
        data_flag = True
    except ValueError:
        try:
            # I will also assume here that the second date given is the return date
            # As a final assumption, we assume that there are only two dates given maximum.
            # Let's first recognize the positions of the dates in the string.
            matcher = Matcher(nlp.vocab)
            matcher.add("DateFinder", None, [{'LOWER':'on'}, {"ENT_TYPE": "DATE"}],
                                            [{'LOWER':'from'}, {"ENT_TYPE": "DATE"}],
                                            [{'LOWER':'to'}, {"ENT_TYPE": "DATE"}],
                                            [{'IS_STOP':True}, {"ENT_TYPE": "DATE"}])
            matches = matcher(nlp(clean_message))
            # First and second date index matchings
            d1 = matches[0]
            d2 = matches[1]
            # Go from first part of date, to one word after (This gets the year, unless there is none, in which case it gets a filler word)
            first_date = nlp(message)[d1[1]:d1[2]+2]
            # If the message ends with the second date, then we need to be careful to not pick up an error
            if (d2[2]==len(clean_message) or d2[2]+1==len(clean_message) or d2[2]+2==len(clean_message)):
                second_date = nlp(clean_message)[d2[1]:]
            else:
                second_date = nlp(clean_message)[d2[1]:d2[2]+2]
                depart_date = dparser.parse(str(first_date),fuzzy=True)
                return_date = dparser.parse(str(second_date),fuzzy=True)
                data_flag=True
        except: 
            return_date = None
            depart_date = None
            data_flag = False
    
    dates = (depart_date, return_date, data_flag)

    return dates

def get_locs(message):
    # Remove punctuation
    clean_message = re.sub(r'[^\w\s]','',message)

    from_matcher = Matcher(nlp.vocab)
    from_matcher.add("LocFinder", None, [{'LOWER':'from'},{'ENT_TYPE':'GPE'}],
                                        [{'LOWER':'of'},{'ENT_TYPE':'GPE'}])
    to_matcher = Matcher(nlp.vocab)
    to_matcher.add("LocFinder", None, [{'LOWER':'to'},{'ENT_TYPE':'GPE'}],
                                        [{'LOWER':'visit'},{'ENT_TYPE':'GPE'}],
                                        [{'LOWER':'see'},{'ENT_TYPE':'GPE'}],
                                        [{'LOWER':'out'},{'ENT_TYPE':'GPE'}],
                                        [{'LOWER':'travel'},{'ENT_TYPE':'GPE'}])
    from_match = from_matcher(nlp(clean_message))
    to_match = to_matcher(nlp(clean_message))

    if len(from_match)>0:
        loc1 = from_match[0]
        origin = str(nlp(clean_message)[loc1[1]+1])
    else:
        origin = None
    if len(to_match)>0:
        loc2 = to_match[0]
        dest = str(nlp(clean_message)[loc2[1]+1])
    else:
        dest = None
    
    if (origin == None and dest == None):
        data_flag = False
    else:
        data_flag = True
    
    locations = (str(origin), str(dest), data_flag)

    return locations

def get_budget(message):
    budget_matcher = Matcher(nlp.vocab)
    budget_matcher.add('BudgetFinder', None, [{'LOWER':'of'},{'IS_DIGIT':True}],
                                            [{'LOWER':'spend'},{'IS_DIGIT':True}],
                                            [{'LOWER':'under'},{'IS_DIGIT':True}],
                                            [{'LOWER':'than'},{'IS_DIGIT':True}],
                                            [{'ENT_TYPE':'MONEY'}],
                                            [{'ORTH':'$'},{'IS_DIGIT':True}])
    match = budget_matcher(nlp(message))
    try:
        budget = nlp(message)[match[0][2]-1]
    except IndexError:
        budget = None

    if budget == None:
        data_flag=False
    else:
        data_flag = True
    return (str(budget), data_flag)

city_code = pd.read_csv(f'Airports.csv').groupby('Location').max()
code_dict = city_code.to_dict()['Code']

def iata_code_lookup(city, codes, origin_city=True):
    '''
    This function takes in a city name as well as a dictionary with city names and their IATA codes. It then checks whether this is the origin city or not.
    If the city is in the dictionary, it will output the code. Otherwise, it will output an empty string and an error message that depends on whether or not the city was the origin or destination.
    '''
    try: 
        iata_code = codes[city]
    except KeyError:
        iata_code = ''
        if origin_city:
            print(f'Unfortunately there are no flights leaving {city} at this time.')
            return None
        else:
            print(f'Unfortunately there are no flights to {city} at this time.')
            return None
    return iata_code

def get_flight(flight_info, codes):
    amadeus = Client(
        client_id=os.getenv("AMADEUS_API_KEY"),
        client_secret=os.getenv("AMADEUS_SECRET")
    )

    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode = iata_code_lookup(flight_info['origin_loc'], codes, origin_city=True),
            destinationLocationCode = iata_code_lookup(flight_info['dest_loc'], codes, origin_city=False),
            departureDate=f"{flight_info['depart_date'].year}-{str(flight_info['depart_date'].month) if int(flight_info['depart_date'].month)>9 else '0'+str(flight_info['depart_date'].month)}-{str(flight_info['depart_date'].day) if int(flight_info['depart_date'].day)>9 else '0'+str(flight_info['depart_date'].day)}",
            returnDate=f"{flight_info['return_date'].year}-{str(flight_info['return_date'].month) if int(flight_info['return_date'].month)>9 else '0'+str(flight_info['return_date'].month)}-{str(flight_info['return_date'].day) if int(flight_info['return_date'].day)>9 else '0'+str(flight_info['return_date'].day)}",
            maxPrice=int(flight_info['budget']),
            currencyCode='CAD',
            adults=1)
        print(f"Your flight has been booked! Here are the details, tickets will be sent to you by email.")
        print(f"Departure Time: {response.data[0]['itineraries'][0]['segments'][0]['departure']['at']}") 
        print(f"Arrival Time: {response.data[0]['itineraries'][0]['segments'][0]['arrival']['at']}")
        print(f"Total Price: {response.data[0]['price']['total']}")
        print(f"Flight Class: {response.data[0]['travelerPricings'][0]['fareDetailsBySegment'][0]['cabin']}")
    except ResponseError as error:
        print(error)
    return None

  # Now that the chatterbot is trained reasonably, let's create the structure that loops through the conversation, until it has enough information to book a flight.
def Traveler():
    '''
    Due to time constraints, I'm going to let the chatbot do its chatbot thing until it recognizes that a flight is being booked. At that point I'll kick in the script and argue that this is what a 
    human salesman would do...
    '''
    # this list of text queues the chatbot to quit
    byes = ['Goodbye', 'Good bye','You as well, goodbye','Later', 'Bye for now','Bye','See ya','Have a good night','Farewell','Goodbye!', 'bye']
    # First, we'll instantiate a dictionary for the values we want to populate.
    flight_info = {'depart_date':0,
                    'return_date':0,
                    'origin_loc':0,
                    'dest_loc':0,
                    'budget':0
                    }
    conversation_ongoing = True

    while conversation_ongoing:
        user_input = input()
        # If user says something that queues a goodbye message from the chatbot, it gets the chance to say that, and then the loop ends.
        if any(bye in user_input for bye in byes):
            print(random.choice(byes))
            conversation_ongoing = False
            break
        # If there is no information in the travel dictionary, then the chatbot should hold the conversation
        if not ((get_locs(user_input)[2]) or (get_dates(user_input)[2]) or (get_budget(user_input)[1])):
            # Get chatbot response
            print(chatbot.get_response(user_input))
        
        else:
            # Get Locations, if any
            if ((flight_info['dest_loc']==0) and (get_locs(user_input)==None)):
                response_list_dest_loc = ['Where did you want to go?', 'Where were you thinking of travelling to?', 'Where to?']
                print(random.choice(response_list_dest_loc))
                user_input = input()
                locations = get_locs(user_input)
                flight_info['dest_loc']=locations[1]
            else:
                flight_info['dest_loc']=get_locs(user_input)[1]

            if flight_info['origin_loc']==0:
                response_list_origin_loc = ['Where are you leaving from?', 'What city are you flying out of?', 'Where ya coming from?']
                print(random.choice(response_list_origin_loc))
                user_input = input()
                locations = get_locs(user_input)
                flight_info['origin_loc']=locations[0]

            # Get Dates, if any
            if flight_info['depart_date']==0:
                response_list_depart_date = ['When did you want to leave?', 'What dates were you thinking?', 'When is your travel date?', 'For what days should I book the flight?']
                print(random.choice(response_list_depart_date))
                user_input=input()
                dates = get_dates(user_input)
                flight_info['depart_date']=dates[0]

            if flight_info['return_date']==0:
                response_list_return_date = ['When did you want to come back?', 'When do you need to return?', 'What should I set for return date?']
                print(random.choice(response_list_return_date))
                user_input=input()
                dates = get_dates(user_input)
                flight_info['return_date']=dates[0]

            # Get budget if any
            if flight_info['budget']==0:
                response_list_budget = ['How much are you willing to spend on this trip?', "What's your budget for this trip?", "Do you have a price limit for the trip?"]
                print(random.choice(response_list_budget))
                user_input = input()
                budget = get_budget(user_input)
                flight_info['budget'] = budget[0]
            
            if all(flight_info[key] != None for key in flight_info.keys()):
                print(f"So to recap, you want to book a trip from {flight_info['origin_loc']} to {flight_info['dest_loc']} for {flight_info['depart_date']} to {flight_info['return_date']} and you have a budget of {flight_info['budget']}. If this is correct, please say yes, if not, say no")
                user_input = input()
                if (user_input.lower() in ['y', 'yes','yeah', 'yep']):
                    print("Excellent, let's, see what the system says. Loading...")
                    get_flight(flight_info, code_dict)
                else:
                    print("Oh, well thats too bad. Better luck next time!")
                    break

        # get number of people, maybe later

    return None
print('Ready')
Traveler()