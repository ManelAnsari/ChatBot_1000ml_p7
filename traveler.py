path='data/'

import discord
from discord.ext import commands
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

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_SERVER')

# Setting up the discord api client.
client = discord.Client()
bot = commands.Bot(command_prefix = '!')

@client.event
# This is the function that discord calls when it starts.
async def on_ready():
    guild = discord.utils.get(client.guilds, name = GUILD)
    
    # as this connects, it prints out what guild/server it is connected to
    print(
        f'{client.user} is connected to the following guild:\n'
        f'{guild.name}(id:{guild.id})')

    # now print out a string of the members on the server/guild
    members = '\n - '.join([member.name for member in guild.members])
    print(f'Guild Members:\n - {members}')

@client.event
async def on_member_join(member):
    # these functions wait until all other members of the coroutine are finished exucuting.
    join_messages = [
        f'Hey {member.name}, welcome to the Traveler\'s Guild, how can I help you?',
        f'Nice to meet you, {member.name}, I am the Traveler! What can I do for you today?',
        f'Well hello there, {member.name}! What can I help you with?'
    ]

    response = random.choice(join_messages)
    await member.create_dm()
    await member.dm_channel.send(response)

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
chatbot = ChatBot('Traveler',logic_adapters=["chatterbot.logic.BestMatch"])
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

def date_from_matcher(tokenized):
    matcher = Matcher(nlp.vocab)
    matcher.add("DateFromFinder", None, [{'LOWER':'leave'}, {'LOWER':'on'}, {"ENT_TYPE": "DATE"}],
                                        [{'LOWER':'leaving'}, {'LOWER':'on'}, {"ENT_TYPE": "DATE"}],
                                        [{'LOWER':'leave'}, {"ENT_TYPE": "DATE"}],
                                        [{'LOWER':'leaving'}, {"ENT_TYPE": "DATE"}],
                                        [{'LOWER':'go'}, {'LOWER':'on'}, {"ENT_TYPE": "DATE"}],
                                        [{'LOWER':'on'}, {"ENT_TYPE": "DATE"}], 
                                        [{'LOWER':'from'}, {"ENT_TYPE": "DATE"}])
    return matcher(tokenized)

    
def date_to_matcher(tokenized):
    matcher = Matcher(nlp.vocab)
    matcher.add("DateToFinder", None, [{'LOWER':'return'},{'LOWER':'on'}, {"ENT_TYPE": "DATE"}],
                                      [{'LOWER':'returning'},{'LOWER':'on'}, {"ENT_TYPE": "DATE"}],
                                      [{'LOWER':'return'},{"ENT_TYPE": "DATE"}],
                                      [{'LOWER':'returning'}, {"ENT_TYPE": "DATE"}],
                                      [{'LOWER':'back'},{'LOWER':'on'}, {"ENT_TYPE": "DATE"}],
                                      [{'LOWER':'to'}, {"ENT_TYPE": "DATE"}])
    return matcher(tokenized)
   

def origin_matcher(tokenized):
    matcher = Matcher(nlp.vocab)
    matcher.add("OriginFinder", None, [{'LOWER':'from'},{'ENT_TYPE':'GPE'}],
                                      [{'LOWER':'of'},{'ENT_TYPE':'GPE'}],
                                      [{'LOWER':'leaving'},{'ENT_TYPE':'GPE'}],
                                      [{'LOWER':'leave'},{'ENT_TYPE':'GPE'}])
    return matcher(tokenized)

def dest_matcher(tokenized):
    matcher = Matcher(nlp.vocab)
    matcher.add("DestinationFinder", None, [{'LOWER':'to'},{'ENT_TYPE':'GPE'}],
                                           [{'LOWER':'visit'},{'ENT_TYPE':'GPE'}],
                                           [{'LOWER':'see'},{'ENT_TYPE':'GPE'}],
                                           [{'LOWER':'out'},{'ENT_TYPE':'GPE'}],
                                           [{'LOWER':'travel'},{'ENT_TYPE':'GPE'}])
    return matcher(tokenized)

def budget_matcher(tokenized):
    matcher = Matcher(nlp.vocab)
    matcher.add('BudgetFinder', None, [{'LOWER':'of'},{'IS_DIGIT':True}],
                                      [{'LOWER':'spend'},{'IS_DIGIT':True}],
                                      [{'LOWER':'under'},{'IS_DIGIT':True}],
                                      [{'LOWER':'than'},{'IS_DIGIT':True}],
                                      [{'ENT_TYPE':'MONEY'}],
                                      [{'ORTH':'$'},{'IS_DIGIT':True}])
    return matcher(tokenized)

def entity_looker(message):
    '''
    This function looks at a message from the user and seeks for matches in dates, locations and budget
    and outputs a value of True if they are contained in the message. 
    '''
    
    # Tokenize the sentence for entity matching.
    tokens = nlp(message)
    # This portion uses the boolean nature of an empty list. If the list is empty, we get False, else True
    # Date From Match
    DF_match = not not date_from_matcher(tokens)
    
    # Date To Match
    DT_match = not not date_to_matcher(tokens)
    
    # Origin Match
    O_match = not not origin_matcher(tokens)
    
    # Destination Match
    D_match = not not dest_matcher(tokens)
    
    # Budget Match
    B_match = not not budget_matcher(tokens)
    
    # This returns a tuple of the Trues/Falses, so the functions know what they are looking for specifically.
    return (DF_match, DT_match, O_match, D_match, B_match)

def entity_picker(message):
    '''
    This function takes the match information, and calls on the functions necessary to extract the info
    from the message. It then converts the entities that it picked out to an API readable form.
    It then returns a set of the entities. If there were no matches, it returns None.
    '''
    
    matches = entity_looker(message)
    from_date = to_date = origin = dest = budget_int = 0
    
    # From Date match
    if matches[0]:
        # Get the from date (comes back as DateTime)
        from_date_dt = get_from_dates(message)
        # Convert dates for API
        from_date = convert_date(from_date_dt)
    
    # To Date match
    if matches[1]:
        # Get the from date (comes back as DateTime)
        to_date_dt = get_to_dates(message)
        # Convert dates for API
        to_date = convert_date(to_date_dt)
        
    # Origin Location match
    if matches[2]:
        # Get the origin location
        origin = get_origin(message)
    
    # Destination Location match
    if matches[3]:
        # get the destination location
        dest = get_dest(message)
    
    # Budget match
    if matches[4]:
        # get the budget
        budget = get_budget(message)
        # convert to int
        budget_int = int(budget)
    
    return (from_date, to_date, origin, dest, budget_int)

def get_from_dates(message):
    '''
    This function is designed to extract a datetime object from the departure date, called "from_date". 
    If there are two dates in the message, I will _assume_ that the first date is the departure date, and the
    second is the return date. 
    '''
    # Tokenize
    tokens = nlp(message)
    # Extract entities
    entities = [(X.label_, X.text) for X in tokens.ents]
    # get rid of anything not a date
    dates = [x for x in entities if x[0]=='DATE']

    # Next, we will look to see if someone used the words "tomorrow", "next week" or "today"
    text = [x[1].lower() for x in dates]
    today = datetime.datetime.today()
    
    if len(dates)>1:
        try: 
            # we check if these words are in the message, and if so, do dates accordingly
            if any(day in text for day in ['today','tomorrow','next week']):
                from_date = word_dates(text[0], today)
            else: 
                from_date = dparser.parse(dates[0][1],fuzzy=True)
        except:
            # Need a thing here to communicate with discord, that states this is not a date type that is known
            return None
    elif len(dates)==1:
        try:
            if any(day in text for day in ['today','tomorrow','next week']):
                from_date = word_dates(text[0], today)
            else:
                from_date = dparser.parse(dates[0][1],fuzzy=True)
        except:
            # Communicate with discord again
            return None
        
    return from_date

def word_dates(date_text, today):
    if 'today' in date_text:
        date = today
    elif 'tomorrow' in date_text:
        date = today + datetime.timedelta(days=1)
    elif 'next week' in date_text:
        date = today + datetime.timedelta(days=7)
    else:
        date = None
    return date

def convert_date(dtime):
    '''
    This function converts a date time object to a month, day, year. This is required for the API
    '''
    YYYY = str(dtime.year)
    MM = str(dtime.month) if int(dtime.month) >9 else '0'+str(dtime.month)
    DD = str(dtime.day) if int(dtime.day)>9 else '0'+str(dtime.day)
    return f"{YYYY}-{MM}-{DD}"

def get_to_dates(message):
    '''
    This function is designed to extract a datetime object from the Return date, called "to_date". 
    If there are two dates in the message, I will _assume_ that the first date is the departure date, and the
    second is the return date. 
    '''
    # Tokenize
    tokens = nlp(message)
    # Extract entities
    entities = [(X.label_, X.text) for X in tokens.ents]
    # get rid of anything not a date
    dates = [x for x in entities if x[0]=='DATE']
    
    # Next, we will look to see if someone used the words "tomorrow", "next week" or "today"
    text = [x[1].lower() for x in dates]
    today = datetime.datetime.today()
    
    if len(dates)>1:
        try: 
            # we check if these words are in the message, and if so, do dates accordingly
            if any(day in text for day in ['today','tomorrow','next week']):
                to_date = word_dates(text[1], today)
            else: 
                to_date = dparser.parse(dates[1][1], fuzzy=True)
        except:
            # Need a thing here to communicate with discord, that states this is not a date type that is known
            return None
    elif len(dates)==1:
        try:
            if any(day in text for day in ['today','tomorrow','next week']):
                to_date = word_dates(text[0], today)
            else:
                to_date = dparser.parse(dates[0][1],fuzzy=True)
        except:
            # Communicate with discord again
            return None
        
    return to_date

def get_origin(message):
    
    # Tokenize
    tokens = nlp(message)
    # Extract entities
    entities = [(X.label_, X.text) for X in tokens.ents]
    # get rid of anything not a date
    loc_ent = [x for x in entities if x[0]=='GPE']
    locs = [x[1].lower() for x in loc_ent]
    
    if len(locs)>1:
        # If there is more than one location in the string, we just need to figure out what order those locations
        # are in, in the sentence, which this logic does.
        if dest_matcher(tokens)[0][2] > origin_matcher(tokens)[0][2]:
            origin = locs[0]
        else:
            origin = locs[1]
    else:
        origin = locs[0]
        
    return origin

def get_dest(message):
    
    # Tokenize
    tokens = nlp(message)
    # Extract entities
    entities = [(X.label_, X.text) for X in tokens.ents]
    # get rid of anything not a date
    loc_ent = [x for x in entities if x[0]=='GPE']
    locs = [x[1].lower() for x in loc_ent]
    
    if len(locs)>1:
        # If there is more than one location in the string, we just need to figure out what order those locations
        # are in, in the sentence, which this logic does.
        if dest_matcher(tokens)[0][2] > origin_matcher(tokens)[0][2]:
            dest = locs[1]
        else:
            dest = locs[0]
    else:
        dest = locs[0]
        
    return dest

def get_budget(message):
    tokens = nlp(message)
    budget = budget_matcher(tokens)
    match = budget_matcher(nlp(message))
    try:
        budget = nlp(message)[match[0][2]-1]
    except IndexError:
        budget = None

    return int(str(budget))

# Read in IATA code csv
city_code = pd.read_csv('Airports.csv')
# make everything lowercase to account for user 
city_code['Location'] = city_code['Location'].str.lower()
# Make the location the index
city_code = city_code.groupby('Location').max()
# Turn into dictionary where the cities are the keys and codes are the values.
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
            departureDate = flight_info['depart_date'],
            returnDate = flight_info['return_date'],
            maxPrice = flight_info['budget'],
            currencyCode='CAD',
            adults=1)
        result = f'''Your flight has been booked! Here are the details, tickets will be sent to you by email.
        Departure Time: {response.data[0]['itineraries'][0]['segments'][0]['departure']['at']}
        Arrival Time: {response.data[0]['itineraries'][0]['segments'][0]['arrival']['at']}
        Total Price: {response.data[0]['price']['total']}
        Flight Class: {response.data[0]['travelerPricings'][0]['fareDetailsBySegment'][0]['cabin']}
        '''
    except ResponseError as error:
        result = error
    return result

flight_info = {'depart_date':0,
               'return_date':0,
               'origin_loc':0,
               'dest_loc':0,
               'budget':0
              }

def Traveler(message):  # sourcery skip: inline-immediately-returned-variable
    # First, if the message contains a confirmation, we should continue with the API booking
    if ('confirmed' in message) or ('Confirmed' in message):
        return get_flight(flight_info, code_dict)
    # See what matches we have:
    values = entity_picker(message)
    
    if values[0] != 0:
        flight_info['depart_date'] = values[0]
        
    if values[1] != 0:
        flight_info['return_date'] = values[1]
        
    if values[2] != 0:
        flight_info['origin_loc'] = values[2]
        
    if values[3] != 0:
        flight_info['dest_loc'] = values[3]
        
    if values[4] != 0:
        flight_info['budget'] = values[4]
    
    constraint_bool = [vals!=0 for vals in flight_info.values()]
    if (any(constraint_bool)) and not (all(constraint_bool)):
        if flight_info['dest_loc'] == 0:
            response_list_dest_loc = ['Where did you want to go?', 
                                      'Where were you thinking of travelling to?', 
                                      'What is your desired destination?']
            return random.choice(response_list_dest_loc)
        if flight_info['origin_loc'] == 0: 
            response_list_origin_loc = ['Where are you leaving from?', 
                                        'What city are you flying out of?', 
                                        'Where will you be flying from?']
            return random.choice(response_list_origin_loc)
        if flight_info['depart_date'] == 0:
            response_list_depart_date = ['When did you want to leave?', 
                                         'What dates were you thinking?', 
                                         'When is your travel date?', 
                                         'For what days should I book the flight?']
            return random.choice(response_list_depart_date)
        if flight_info['return_date'] == 0:
            response_list_return_date = ['When did you want to come back?', 
                                         'When do you need to return?', 
                                         'What should I set for return date?']
            return random.choice(response_list_return_date)
        if flight_info['budget'] == 0:
            response_list_budget = ['How much are you willing to spend on this trip?', 
                                    "What's your budget for this trip?", 
                                    "Do you have a price limit for the trip?"]
            return random.choice(response_list_budget)
    
    elif all(constraint_bool):
        finalize= f"Awesome! Looks like I have everything I need to book your flight! Let me just confirm with you. You are booking a flight to {flight_info['dest_loc'].upper()}, and you are flying from {flight_info['origin_loc'].upper()}. You will be leaving on {flight_info['depart_date']} and returning on {flight_info['return_date']}. Finally, your budget is ${flight_info['budget']}. Is all this information correct? If yes, please type 'Confirmed'. If this is not correct, please provide the correct info in detail."
        return finalize
    
    else:
        return chatbot.get_response(message)

@client.event
async def on_message(message):
    if message.author == client.user:
        # This is so that the bot never responds to its own message
        return
    # These statements will redirect
    redirect_statements = [
        'Let\'s chat in private, I may need to ask for your personal information!',
        'Can I talk to you over here? I\'ll probably need some personal information!',
        f'Hey {message.author.name} let\'s use direct messaging for privacy.'
    ]

    if ((client.user.mentioned_in(message)) and (str(message.channel) =='travel-booking')):
        response = random.choice(redirect_statements)
        await message.author.create_dm()
        await message.author.dm_channel.send(response)

    if (str(message.channel) != 'travel-booking'):
        bot_response = Traveler(str(message.content))
        print(str(message.content))
        print(bot_response)
        await message.channel.send(bot_response)

client.run(TOKEN)