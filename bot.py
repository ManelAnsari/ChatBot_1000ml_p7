import os
import random

import discord
from dotenv import load_dotenv
from discord.ext import commands

# Import chatbot file
#from traveler import *

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

@bot.command(name='create-channel')
@commands.has_role('admin')
async def create_channel(ctx, channel_name='new-channel'):
    guild = ctx.guild
    existing_channel = discord.utils.get(guild.channels, name=channel_name)
    if not existing_channel:
        print(f'Creating a new channel: {channel_name}')
        await guild.create_text_channel(channel_name)

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.errors.CheckFailure):
        await ctx.send('You do not have the correct role for this command')

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
    elif message.content == 'raise-exception':
        raise discord.DiscordException

@client.event
async def on_error(event, *args, **kwargs):
    with open('err.log', 'a') as f:
        if event == 'on_message':
            f.write(f'Unhandled message: {args[0]}\n')
        else:
            raise

client.run(TOKEN)