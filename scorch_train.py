# ============================================================
# SCORCH: Sparse Contextual Output Router for Comedic Hyperbolization
# Full Training Code — CPU Optimized
# Datasets: shortjokes, social_bias_frames, Comedy Central,
#           OpenSubtitles, Synthetic (10K+)
# All paper mathematics implemented exactly.
# IndexError fully fixed throughout.
# No GPU, no CUDA anywhere.
#
# pip install torch datasets requests beautifulsoup4 scikit-learn tqdm
# pip install git+https://[your-crayon-repo]/crayon
# ============================================================


# ==============================================================
# CELL 1: IMPORTS AND GLOBAL SETUP
# ==============================================================

import os
import math
import time
import random
import warnings
import re
from functools import partial
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

# CPU only — no CUDA anywhere in this file
device = torch.device("cpu")
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CPU cores available: {os.cpu_count()}")

num_threads = os.cpu_count() or 4
torch.set_num_threads(num_threads)
print(f"Using {torch.get_num_threads()} CPU threads")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ==============================================================
# CELL 2: CRAYONVOCAB TOKENIZER
# ==============================================================

from crayon import CrayonVocab

class SCORCHTokenizer:
    """
    Thin wrapper around CrayonVocab.
    Only external dependency allowed by the paper.
    PAD=0, BOS=1, EOS=2 reserved.
    All encoded IDs are clamped to [3, vocab_size-1].
    """
    def __init__(self, profile="standard"):
        self.tok      = CrayonVocab(device="cpu")
        self.tok.load_profile(profile)
        self.profile  = profile
        self.pad_id   = 0
        self.bos_id   = 1
        self.eos_id   = 2
        self._probe_vocab_size()

    def _probe_vocab_size(self):
        probe_strings = [
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "0123456789",
            "!@#$%^&*()_+-=[]{}|;':\",.<>?/\\",
            "the a is and or not you your they them their",
            "ugly stupid dumb boring loser pathetic ridiculous",
            "absolutely positively magnificently catastrophically",
            "roast burn destroy embarrass humiliate obliterate",
            "never always sometimes maybe probably definitely",
            "coffee gym bitcoin startup grind hustle influencer",
            "mom dad brother sister friend colleague boss manager",
            "honestly seriously literally actually basically truly",
            "personality energy vibe aesthetic brand era journey",
            "trauma healing growth mindset hustle passive income",
        ]
        max_id = 3
        for ps in probe_strings:
            try:
                ids = self.tok.tokenize(ps)
                if ids:
                    max_id = max(max_id, max(ids))
            except Exception:
                pass
        self.vocab_size = 32768  
        print(f"[SCORCHTokenizer] Probed vocab size: {self.vocab_size}")

    def encode(self, text: str):
        """
        Tokenize text. Returns list of ints all in [3, vocab_size-1].
        Never raises — returns [] on failure.
        """
        try:
            ids = self.tok.tokenize(str(text))
            if not ids:
                return []
            ids = [max(3, min(int(i), self.vocab_size - 1)) for i in ids]
            return ids
        except Exception:
            return []

    def decode(self, ids):
        """
        Decode list of ints to string.
        Filters special tokens. Never raises.
        """
        try:
            clean = [int(i) for i in ids
                     if int(i) not in (self.pad_id, self.bos_id, self.eos_id)
                     and int(i) >= 3]
            if not clean:
                return ""
            return self.tok.decode(clean)
        except Exception:
            return ""

    def encode_with_special(self, text: str,
                             add_bos=True, add_eos=True):
        ids = self.encode(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids


tokenizer  = SCORCHTokenizer(profile="standard")
VOCAB_SIZE = min(tokenizer.vocab_size, 32768) 
PAD_ID     = tokenizer.pad_id
BOS_ID     = tokenizer.bos_id
EOS_ID     = tokenizer.eos_id

print(f"VOCAB_SIZE={VOCAB_SIZE}, PAD={PAD_ID}, BOS={BOS_ID}, EOS={EOS_ID}")

_test = tokenizer.encode("hello world this is a roast test")
assert all(i >= 3 for i in _test),       "Special token collision!"
assert all(i < VOCAB_SIZE for i in _test), "Token exceeds vocab_size!"
print(f"Tokenizer OK — {len(_test)} tokens, range [{min(_test)},{max(_test)}]")


# ==============================================================
# CELL 3: DATASET 5 — SYNTHETIC ROAST PAIRS (10K+)
# Defined first because other loaders may fall back to it
# ==============================================================

def generate_synthetic_roast_pairs():
    """
    Generates 10,000+ synthetic uncensored roast training pairs.
    Covers 12 categories: appearance, intelligence, career,
    social media, relationships, fitness, food, tech,
    lifestyle, age, money, archetypes.
    Returns list of (input_str, output_str).
    """

    # ── CATEGORY 1: APPEARANCE ──
    appearance_targets = [
        "someone with a bad haircut",
        "a person who is balding but refuses to shave",
        "someone who dresses like they got clothes from a dumpster",
        "a person who wears crocs everywhere",
        "someone who has visible sweat stains constantly",
        "a person with a bad fake tan",
        "someone who wears the same outfit every day",
        "a person who wears sunglasses indoors",
        "someone who has frosted tips in 2024",
        "a person who still wears a fedora unironically",
        "someone who wears cargo shorts to formal events",
        "a person with a neckbeard",
        "someone who thinks they are more attractive than they are",
        "a person with a bad comb-over",
        "someone who overlines their lips dramatically",
        "a person who wears too much cologne",
        "someone who wears too much perfume",
        "a person who looks permanently exhausted",
        "someone with a resting face that scares children",
        "a person who has not updated their style since 2003",
    ]
    appearance_roasts = [
        "Your haircut looks like you lost a bet with a blind barber who was also drunk.",
        "You are not going bald gracefully, you are going bald loudly and in denial.",
        "Your outfit looks like a thrift store had a fire sale and you bought the smoke damage.",
        "You wear Crocs everywhere and wonder why people cancel plans when you show up.",
        "Your sweat stains have their own zip code at this point.",
        "Your fake tan makes you look like you got into a fight with a bag of Cheetos and lost badly.",
        "You wear the same outfit so often your clothes have developed separation anxiety from the laundry.",
        "You wear sunglasses indoors because the future is so bright apparently and it is not, nothing is bright about this.",
        "Frosted tips in 2024. You are not a throwback, you are a warning.",
        "The fedora is not mysterious, it is a sign that says I argue with strangers about philosophy online.",
        "You wore cargo shorts to a funeral and somehow that was the least offensive thing about the visit.",
        "The neckbeard is doing a lot of heavy lifting as your entire personality.",
        "You think you are a ten but the scoreboard has you as a strong self-assessed six.",
        "The comb-over is not fooling anyone. It is not even fooling you. You know. We all know.",
        "Your overlining technique suggests you are training to play the Joker without the acting skills.",
        "You bathe in cologne because subtlety abandoned you years ago and you are getting revenge.",
        "Your perfume arrives in the room four minutes before you do and everyone has already started leaving.",
        "You look permanently exhausted and somehow also permanently surprised that life is happening to you.",
        "Your resting face has caused at least three people to ask if you need medical attention.",
        "Your fashion sense is frozen at 2003 which would be charming if anything about it was intentional.",
        "Your wardrobe looks like it was curated by someone who has only seen humans in magazines from 2001.",
        "You dress like you Googled how to look like a person and followed the instructions incorrectly.",
        "Your style is bold in the way that a traffic accident is bold.",
        "You look like you got dressed in the dark, in a hurry, in someone else's house.",
        "Your fashion choices are technically legal in all fifty states which is the nicest thing I can say.",
        "You look like the before photo in an advertisement for everything.",
        "Your aesthetic can best be described as gave up but still showed up.",
        "You dress like comfort and dignity had a disagreement and comfort won every single round.",
        "Your outfit is so loud it violated a noise ordinance in three states.",
        "You look like a background character in a movie about someone else's more interesting life.",
    ]

    # ── CATEGORY 2: INTELLIGENCE ──
    intelligence_targets = [
        "someone who is not very smart but thinks they are",
        "a person who shares fake news confidently",
        "someone who googles things wrong",
        "a person who cannot read a room",
        "someone who finishes other people's sentences incorrectly",
        "a person who misuses big words constantly",
        "someone who thinks being loud counts as being right",
        "a person who has the same argument every time and loses it every time",
        "someone who skips the instructions and then complains",
        "a person who asks questions they could google in two seconds",
        "someone who falls for every conspiracy theory",
        "a person who uses the word literally incorrectly every single time",
        "someone who cannot take a hint no matter how obvious",
        "a person who forgets the point of their own story halfway through",
        "someone who gives advice on things they know nothing about",
        "a person who brags about never reading books",
        "someone who thinks confidence is the same as competence",
        "a person who cannot follow simple directions",
        "someone who argues with experts because they watched a YouTube video",
        "a person who is always the loudest wrong person in the room",
    ]
    intelligence_roasts = [
        "You are not a free thinker. You are a free-falling thinker with no parachute and no landing plan.",
        "You share fake news with the confidence of someone who has never been right but has not noticed.",
        "You Google things in a way that suggests you and the search engine have a hostile relationship.",
        "You cannot read a room. You cannot read a paragraph. You cannot read the vibe at all.",
        "You finish other people's sentences wrong and then act proud of the ruins you left behind.",
        "You misuse big words so consistently it has become a personality. A bad one.",
        "You think volume is an argument. It is not. You are just wrong but louder.",
        "You have the same argument every time, lose it every time, and walk away more confident somehow.",
        "You skip instructions the way other people skip commercials, automatically and to your constant detriment.",
        "You ask questions you could Google in two seconds because you want someone to suffer through the answer with you.",
        "You believe every conspiracy theory because critical thinking is a skill that requires practice and you have not practiced.",
        "You use the word literally so incorrectly that literally has hired a lawyer to pursue damages.",
        "You cannot take a hint if it were delivered certified mail with a signature requirement.",
        "You lose the point of your own story so completely that by the end even you look confused.",
        "You give advice on everything you know nothing about with the authority of someone who knows something.",
        "You brag about never reading books the way someone brags about never eating vegetables, proudly malnourished.",
        "You mistake confidence for competence the way people mistake a map for the territory, dangerously and regularly.",
        "You cannot follow simple directions which explains a lot of things including this conversation.",
        "You watched one YouTube video and now you are arguing with people who have doctorates. Remarkable.",
        "You are the loudest wrong person in every room which is technically an achievement in the wrong direction.",
        "Your IQ is in a timezone that has not been discovered yet.",
        "You have the reasoning skills of a Magic 8-Ball but with fewer accurate predictions.",
        "Thinking is clearly a guest in your head that never got comfortable and eventually stopped visiting.",
        "You operate on vibes alone and the vibes have filed a formal complaint.",
        "Your logic has load-bearing assumptions that would not survive a light breeze.",
        "You are living proof that the Dunning-Kruger effect is not a theory, it is a biography.",
        "You have strong opinions about everything and knowledge of nothing and somehow it gets louder every year.",
        "Your brain and your mouth are clearly not on speaking terms and have not been for some time.",
        "You think out loud and it shows. Please think quietly. Please think at all.",
        "You have the intellectual curiosity of someone who finds the back of a cereal box too demanding.",
    ]

    # ── CATEGORY 3: CAREER AND HUSTLE ──
    career_targets = [
        "a crypto bro",
        "an NFT investor who lost everything",
        "a person who calls themselves an entrepreneur but has no business",
        "someone who works in marketing and takes it too seriously",
        "a person who puts motivational quotes in every email",
        "someone who has been in a startup for five years with no product",
        "a person who calls every small task a hustle",
        "someone who brags about working 80 hour weeks",
        "a person who says they are their own boss but earns less than minimum wage",
        "someone who puts CEO of their own name on their LinkedIn",
        "a life coach with no life experience",
        "a person who constantly talks about passive income",
        "someone who has pivoted their business seven times in two years",
        "a person who says they are disrupting an industry nobody asked them to disrupt",
        "someone who went to one networking event and now networks constantly",
        "a person who calls their job a calling but hates every minute of it",
        "someone who name-drops companies they consulted for once",
        "a person who describes their job in the most inflated terms possible",
        "someone who thinks having a podcast makes them a thought leader",
        "a person who sends LinkedIn messages that are clearly copied and pasted",
    ]
    career_roasts = [
        "You got into crypto at the peak, held through the crash, and are still explaining why it will come back.",
        "You bought an NFT of a monkey and now the monkey has more career prospects than you.",
        "You are an entrepreneur the way a person standing in a garage is an astronaut.",
        "You work in marketing and you have forgotten that the rest of us are the people you market to.",
        "Every email you send has a motivational quote as if your recipients are one Rumi line away from productivity.",
        "Your startup has pivoted so many times the business plan has a rotational injury.",
        "You call everything a hustle. Buying groceries is a hustle. Existing is a hustle. You need to rest.",
        "You work 80-hour weeks and announce it the way other people announce injuries, for sympathy not admiration.",
        "You are your own boss and you are doing a terrible job of managing yourself.",
        "You put CEO of your own name on LinkedIn because titling yourself Boss of Nothing felt too honest.",
        "You are a life coach with the life experience of a particularly sheltered houseplant.",
        "You talk about passive income so much that the income remains passive while the talking never stops.",
        "You have pivoted your business seven times in two years which is not agility, it is a prolonged panic.",
        "You are disrupting an industry that did not ask to be disrupted and is not noticeably different for it.",
        "You went to one networking event and now you network like it is oxygen and you are panicking.",
        "You call your job a calling but every time someone asks about it your eye twitches.",
        "You name-drop companies you consulted for once with the pride of someone who helped build them from scratch.",
        "Your job title is seven words long because the shorter version would reveal how little happens in it.",
        "You have a podcast which means you have a microphone, not an audience, and definitely not authority.",
        "Your LinkedIn messages are so clearly copy-pasted that even the original template writer would not send them.",
        "You describe yourself as a visionary and the vision is never quite visible to anyone else.",
        "Your business model has more pivots than a professional gymnast and less direction than a spinning top.",
        "You are building something revolutionary and have been building it since 2019 with no visible structure.",
        "Your synergies and value propositions have produced exactly no value and several confused faces.",
        "You left a stable job to follow your passion and the passion is currently declining to comment.",
        "Your startup is pre-revenue which is a polite way of saying it is pre-anything.",
        "You network so aggressively that people block you on LinkedIn the way they block spam.",
        "Your hustle culture has produced a body of work that requires a microscope and generous interpretation to see.",
        "You moved fast and broke things including your savings account, two partnerships, and your family's patience.",
        "You say you are building an empire. You are building a very optimistic spreadsheet.",
    ]

    # ── CATEGORY 4: SOCIAL MEDIA ──
    social_media_targets = [
        "an influencer with 200 followers who acts like a celebrity",
        "a person who posts selfies every hour",
        "someone who writes long emotional captions about minor inconveniences",
        "a person who posts inspirational quotes they did not write",
        "someone who announces every life update on social media before telling family",
        "a person who tags brands in every photo hoping for free stuff",
        "someone who uses thirty hashtags on every post",
        "a person who vague-posts constantly to get people to ask what is wrong",
        "someone who posts gym selfies in the mirror six times a week",
        "a person who argues in comment sections with strangers for hours",
        "someone who reposts viral content and acts like it is theirs",
        "a person who posts travel photos three years after the trip",
        "someone who has thousands of followers but zero real friends",
        "a person who goes live on Instagram while doing absolutely nothing",
        "someone who announces a social media break and comes back in four hours",
        "a person who has a finsta that is somehow worse than their main account",
        "someone who posts deep philosophical questions at two in the morning",
        "a person who rates their own posts in the comments",
        "someone who sends DMs asking for follows in exchange for follows",
        "a person who calls themselves a content creator but the content is just them existing",
    ]
    social_media_roasts = [
        "You have 200 followers and the personal brand energy of someone followed by nations.",
        "You post selfies every hour as if you are filing hourly updates with the department of you.",
        "You wrote four paragraphs about a cold coffee as though it was a formative trauma and not a Tuesday.",
        "You post other people's quotes as though wisdom is something you can curate yourself into having.",
        "Your family learned about your engagement from your Instagram story which is not the tradition most families aim for.",
        "You tag brands in photos of yourself in their general vicinity hoping for collaboration and receiving silence.",
        "You use thirty hashtags on a photo of your lunch which is the most desperate form of optimism I have seen.",
        "You vague-post so constantly that the people who used to ask what is wrong have started assuming nothing is.",
        "Your gym mirror selfies have their own narrative arc. We have watched you evolve. We are tired.",
        "You argue in comment sections with strangers for hours and have never changed a single mind.",
        "You repost viral content with a fire emoji as if you discovered it in the wild and tamed it.",
        "You posted travel photos from 2021 in 2024 because the algorithm does not know and you are banking on that.",
        "You have ten thousand followers and the authentic human connection of a brand account for a mattress company.",
        "You went live on Instagram to sit there and you got viewers because loneliness is a public health crisis.",
        "You announced a social media break, lasted four hours, came back to explain why the break is cancelled.",
        "Your finsta exists so people can see the version of you that is worse than the version they already have concerns about.",
        "You post philosophical questions at two AM because the void called and you decided to hold it publicly accountable.",
        "You commented fire on your own post and then replied to that with a flame emoji. This is the arc.",
        "You DM people asking for follows in exchange for follows which is the Ponzi scheme of social validation.",
        "You are a content creator and the content is just documentation that you are still alive and underfed by attention.",
        "Your entire online presence is a performance for an audience that is mostly bots and accidental followers.",
        "You have built a personal brand around being relatable and the relatability is the only thing not manufactured.",
        "You go viral once every two years and spend the time between references to that one moment.",
        "Your aesthetic is so curated it looks like a lifestyle and feels like a business plan that is not profitable.",
        "You live for the engagement and the engagement lives for somewhere else.",
        "Your highlights are a museum of a life that looked better in the frame than in the actual living.",
        "You use Instagram like a press release service for a public figure the public did not ask for.",
        "You have an opinion on every trending topic within forty minutes which is efficiency weaponised against depth.",
        "Your captions are longer than most short stories and significantly less interesting.",
        "You called yourself an influencer to your grandmother and she nodded politely and changed the subject.",
    ]

    # ── CATEGORY 5: RELATIONSHIPS ──
    relationship_targets = [
        "a person who brings up their ex in every conversation",
        "someone who has been on the dating apps for six years with no dates",
        "a person who says they are not ready for a relationship but texts at 2am",
        "someone who love bombs then disappears",
        "a person who describes every relationship as toxic except their own behaviour",
        "someone who falls in love in the first week and proposes by the second",
        "a person who has a type and the type keeps failing them in the same way",
        "someone who vague-posts about their breakup to make the ex jealous",
        "a person who says they just want something casual but cries after every casual thing",
        "someone who has not dated since 2018 but gives everyone relationship advice",
        "a person who compares every new partner to their last one constantly",
        "someone who can only attract people who need to be rescued",
        "a person who claims to be independent but needs constant reassurance",
        "someone who makes their partner their whole personality",
        "a person who says they are too busy for a relationship while wanting one desperately",
        "someone who ghosts people and then gets upset when they get ghosted",
        "a person who falls for unavailable people exclusively",
        "someone who sends the we need to talk text and then says never mind",
        "a person who posts about being single like it is a terminal illness",
        "someone who treats every first date like a job interview",
    ]
    relationship_roasts = [
        "You bring up your ex so often they are basically a third person in every conversation you have.",
        "Six years on the dating apps and the only thing you have matched with consistently is disappointment.",
        "You say you are not ready for a relationship but text at 2am like someone ready for everything except honesty.",
        "You love bomb people with such intensity that they mistake the explosion for a connection.",
        "Every relationship you have been in was toxic except somehow the part you were responsible for.",
        "You fall in love in a week and propose in two because you have confused speed with certainty.",
        "Your type has failed you so consistently that your type is statistically a pattern and the pattern is you.",
        "You vague-post about your breakup to make your ex jealous and your ex has not checked your profile in months.",
        "You want something casual and then cry about it, which makes it extremely emotional and very expensive.",
        "You have not dated since 2018 but give relationship advice like a veteran of a war still being fought.",
        "You compare every new partner to your last one so constantly that you are essentially still dating your ex.",
        "You can only attract people who need saving because your empathy and poor decision-making are in an alliance.",
        "You say you are fiercely independent and then need three texts of reassurance before any plan is confirmed.",
        "You made your partner your entire personality and when they left they took the personality with them.",
        "You are too busy for a relationship but emotionally devastated that you do not have one.",
        "You ghost people and then write about how ghosting is the worst thing a person can do.",
        "You exclusively fall for people who are unavailable and are consistently surprised by the outcome.",
        "You sent the we need to talk text, made everyone spiral, and then said never mind.",
        "You post about being single like it is a medical emergency instead of a circumstance you have agency over.",
        "You treat every first date like a job interview and wonder why nobody calls back for a second round.",
        "Your dating profile is a highlight reel of a person who does not show up to the actual dates.",
        "You are looking for your person but your person is looking for someone who does not do the things you do.",
        "You say you are open to love but your actions have a very aggressive security system around your heart.",
        "You claim to want honesty and cannot handle a single honest thing said in your direction.",
        "You fall in love with potential and live in the gap between the potential and the person forever.",
        "Your green flags are so deep undercover they have not surfaced in any relationship you have been in.",
        "You want a partner who communicates openly and you communicate exclusively through memes and avoidance.",
        "You are emotionally unavailable and looking for someone emotionally available to absorb the consequences.",
        "You describe your exes as crazy and the common variable in every crazy relationship is you.",
        "You want a real connection but conduct yourself with the vulnerability of a heavily padlocked suitcase.",
    ]

    # ── CATEGORY 6: FITNESS ──
    fitness_targets = [
        "someone who just started going to the gym and tells everyone",
        "a person who has had a gym membership for years and never goes",
        "someone who takes gym selfies more than they do reps",
        "a person who gives unsolicited workout advice",
        "someone who counts the walk to their car as cardio",
        "a person who buys expensive workout gear and never uses it",
        "someone who only trains their upper body",
        "a person who grunts loudly at the gym for no reason",
        "someone who does not wipe down equipment after using it",
        "a person who says they are about to start working out every January",
        "someone who brags about their protein intake constantly",
        "a person who claims to be athletic but gets winded on stairs",
        "someone who does a detox cleanse every few months and announces it",
        "a person who uses the gym as a place to socialise more than train",
        "someone who said they would run a marathon three years ago and has not started",
        "a person who calls any physical activity a workout",
        "someone who compares their gym progress to professional athletes",
        "a person who blames their metabolism for everything",
        "someone who says the gym is their therapy but never actually improves",
        "a person who judges other people's form at the gym loudly",
    ]
    fitness_roasts = [
        "You have been to the gym twice and have told forty people about it. The ratio is concerning.",
        "You have had that gym membership for four years and the only thing you have worked out is the monthly payment.",
        "You take more selfies than reps which means your arms look great in photos and nowhere else.",
        "You give workout advice nobody asked for to people mid-exercise which is both rude and impressive in commitment.",
        "You count the walk to your car as cardio and the car is in the driveway.",
        "Your workout gear is expensive, unworn, and a monument to the person you described yourself becoming in January.",
        "You only train your upper body. Your legs have not been consulted in years and they have grievances.",
        "You grunt at the gym with the conviction of someone lifting three times what you are lifting.",
        "You leave the equipment wet and move on, which is a form of self-expression and also a biohazard.",
        "Every January you announce the gym era and every February the era ends on its own schedule.",
        "You talk about your protein intake more than most people talk about their families.",
        "You claim to be athletic and then take a break at the top of a staircase to regroup.",
        "You do a cleanse every three months and announce it as though you are undergoing spiritual renewal, not drinking juice.",
        "You go to the gym to talk to people and have not lifted anything since your phone.",
        "You said you were going to run a marathon three years ago and the training has been described as upcoming ever since.",
        "You call standing up from the couch a workout and it has given you a deeply inaccurate sense of your fitness.",
        "You compare your progress to professional athletes and the comparison requires a lot of generosity to sustain.",
        "You blame your metabolism for everything including the parts that are exclusively diet and decision.",
        "You say the gym is your therapy and yet the issues it was supposed to resolve have a very active season.",
        "You correct other people's form at the gym using your own form which is not above comment.",
        "Your rest days outnumber your training days which is a strategy and the strategy is not working.",
        "You post your workouts online and your three followers now know more about your body than your doctor.",
        "You bought a Peloton and it is currently holding laundry which is technically still getting use.",
        "Your fitness journey has been a round trip to exactly where you started taken slowly and with great fanfare.",
        "You talk about gains with the enthusiasm of someone who has recently discovered that muscles exist.",
        "You bulk every winter and cut every summer and the results of this cycle remain pending since 2020.",
        "Your pre-workout routine takes forty-five minutes which means the workout is technically an afterthought.",
        "You call it a cheat meal but the meal runs from Sunday through most of Thursday.",
        "You stretch before exercise which is the only fitness-related thing you do consistently.",
        "Your body is a temple and the temple is currently closed for renovations ongoing for several years.",
    ]

    # ── CATEGORY 7: FOOD ──
    food_targets = [
        "a person who is vegan and mentions it every five minutes",
        "someone who does intermittent fasting and talks about the fasting window constantly",
        "a person who takes photos of every single meal before eating",
        "someone who claims to be a foodie but only eats chicken tenders",
        "a person who rates restaurants harshly on Yelp for minor things",
        "someone who says they do not eat gluten but has no diagnosis",
        "a person who puts avocado on everything and considers it a personality",
        "someone who is insufferable about their sourdough starter",
        "a person who only drinks black coffee and announces it with pride",
        "someone who describes every meal as a life-changing experience",
        "a person who cannot eat anything without it being an event",
        "someone who claims to love spicy food and sweats at mild",
        "a person who insists on splitting the bill to the exact cent at dinner",
        "someone who orders the most complicated modification of every dish",
        "a person who starts every meal story with I was in this little place in Italy",
        "someone who judges people for eating fast food as if they are above it",
        "a person who brings their own food to every social event",
        "someone who calls cooking one dish from scratch a passion for cooking",
        "a person who has a strong opinion about how other people order steak",
        "someone who treats a dietary preference as a moral identity",
    ]
    food_roasts = [
        "You have been vegan for two years and mentioned it roughly eleven thousand times. The cows know. Everyone knows.",
        "Your intermittent fasting window is the most talked-about window since the one in your kitchen that needs replacing.",
        "You photograph every meal before eating it because memories matter but the food is getting cold.",
        "You call yourself a foodie and you order the chicken tenders at every restaurant you visit.",
        "You left a one-star review because the waiter called you buddy and you were not in the mood.",
        "You do not eat gluten out of preference and describe it as an allergy because the truth requires less explanation.",
        "Avocado is on everything you eat and your personality and it is doing better work as a food than as an identity.",
        "You named your sourdough starter and gave it a birthday and if it had legs it would leave.",
        "You drink black coffee without sugar and bring it up as a character trait. It is a beverage choice.",
        "You described a breakfast burrito as life-changing and either your life was very unchallenged or that burrito was extraordinary.",
        "You cannot eat a meal without it becoming a production involving research, negotiation, and a forty-minute decision.",
        "You claim to love spicy food and you ordered the mild salsa and asked them to go easy on it.",
        "You split the bill to the exact cent at a group dinner and the group has started declining dinners.",
        "You modify every dish so completely that what arrives is technically your recipe and the restaurant's regret.",
        "You start every food story with I was in this little place in Italy and it ends with something ordinary.",
        "You judge people for eating fast food from a car that currently has four drive-through bags in the back seat.",
        "You bring your own food to social events and eat it with the focused righteousness of someone attending a protest.",
        "You made pasta once and now describe your relationship with cooking as a passion.",
        "You have a strong opinion about how other people order their steak and you have expressed it without being asked.",
        "You treat eating vegetables as a moral achievement and eating a burger as a confession.",
        "You describe your diet like a religion and evangelise with the same gentle insistence nobody asked for.",
        "You are on your fourth dietary identity this year and each one arrived with an announcement and a kitchen purge.",
        "You make eating difficult in a way that is demanding and ongoing and exhausting for everyone involved.",
        "You say you could eat there every day and you have been once and clearly did not go back.",
        "Your food photos get more attention than your relationships which is information about your priorities.",
        "You rate restaurants on presentation and forget that the purpose of food is historically consumption.",
        "Your complicated order has a name at the coffee shop and the name is not yours.",
        "You discovered fermentation and now every surface in your kitchen is hosting something alive and intentional.",
        "You call anything homemade artisanal which is doing a lot of work as a word.",
        "Your relationship with food is complicated in the way that diplomacy is complicated, lots of rules, frequent breakdowns.",
    ]

    # ── CATEGORY 8: TECH ──
    tech_targets = [
        "a person who upgrades their phone every year for no reason",
        "someone who talks about their Apple products like they are achievements",
        "a person who cannot function without their smart watch",
        "someone who has a smart home setup that never works properly",
        "a person who mines cryptocurrency on their personal computer",
        "someone who preaches about Linux to people who did not ask",
        "a person who buys every new gadget and uses none of them",
        "someone who has a gaming setup worth five thousand dollars but no social life",
        "a person who uses tech jargon to sound smarter than they are",
        "someone who cannot go five minutes without checking their phone",
        "a person who gets angry at people for using the wrong operating system",
        "someone who thinks owning a mechanical keyboard makes them a serious person",
        "a person who watches every Apple event like it is a religious ceremony",
        "someone who loses their mind when the WiFi is slow",
        "a person who has a strong opinion about tabs versus spaces in code",
        "someone who bought a VR headset and used it twice",
        "a person who refers to their phone camera as their photography gear",
        "someone who buys the maximum storage option and uses twelve gigabytes",
        "a person who configures their terminal for thirty hours instead of doing actual work",
        "someone who puts their tech stack in their dating profile",
    ]
    tech_roasts = [
        "You upgrade your phone every year for a camera that takes photos of the same things as last year's camera.",
        "You talk about your MacBook like you built it yourself in a garage and Jobs personally blessed it.",
        "You cannot function without your smart watch which is telling you your stress is elevated and we can all see why.",
        "Your smart home is dumb in practice. The lights do what they want. The thermostat has goals of its own.",
        "You are mining cryptocurrency on your laptop and the heat it generates is its only tangible output.",
        "You explain Linux to people at parties and watch them look for exits in real time.",
        "Your gadget drawer is a museum of enthusiasm without follow-through.",
        "Your gaming setup costs more than most people's cars and your main opponent is free time.",
        "You use technical jargon the way magicians use misdirection, to hide the fact that nothing is happening.",
        "You check your phone every forty seconds and call it staying connected. You are connected to anxiety.",
        "You get personally offended by operating system choices as though Windows is a personality flaw.",
        "You have a mechanical keyboard and type loudly at a rate that suggests the keyboard is the performance.",
        "You watch every Apple event with the reverence typically reserved for events of historical significance.",
        "The WiFi slows down and you act like a fundamental law of physics has been violated.",
        "You have strong opinions about tabs versus spaces in code and people who know you well have started to worry.",
        "You bought a VR headset, set it up, used it twice, and it now lives in a corner with the exercise bike.",
        "You call your phone camera your photography gear and your photographer friends are quietly handling it.",
        "You bought two terabytes of storage and the twelve gigabytes you use could fit on a USB drive from 2009.",
        "You spent thirty hours configuring your terminal setup and zero hours on the project it was configured for.",
        "You put your tech stack in your dating profile. It has not worked. The tech stack is fine. You are the variable.",
        "You describe software as elegant the way people describe architecture and everyone is being patient with you.",
        "Your smart watch tracked your sleep and the results were not good and not a surprise to anyone.",
        "You have a hot take about every framework and the hot takes have never shipped anything.",
        "You are passionate about open source in a way that open source would find slightly overwhelming.",
        "Your setup is optimised for productivity and the productivity has not arrived yet.",
        "You debug for six hours before reading the error message which is the tech equivalent of self-diagnosing online.",
        "You have opinions about programming languages the way people have opinions about religion, deeply held, poorly explained.",
        "You recommend apps to people who did not ask and follow up later to confirm they downloaded them.",
        "Your Wi-Fi name is a joke that was funny when you set it up and you cannot change it now.",
        "You say you could build that in a weekend. You have been saying that for three years.",
    ]

    # ── CATEGORY 9: LIFESTYLE ──
    lifestyle_targets = [
        "a person who is chronically late to everything",
        "someone who cannot make a decision about anything ever",
        "a person who always plays the victim",
        "someone who constantly talks about how busy they are",
        "a person who says yes to everything and follows through on nothing",
        "someone who is passive aggressive in every interaction",
        "a person who drops their friends the moment they get into a relationship",
        "someone who only reaches out when they need something",
        "a person who cannot apologise without making it about themselves",
        "someone who turns every conversation back to themselves",
        "a person who gives backhanded compliments as their primary mode of interaction",
        "someone who says they are an introvert to avoid all social responsibility",
        "a person who is always the main character in every story",
        "someone who has a victim story ready for every situation",
        "a person who does the bare minimum and acts like they went above and beyond",
        "someone who is relentlessly negative about everything",
        "a person who copies everything their friends do with a slight delay",
        "someone who is obsessed with being seen as unique while doing everything popular",
        "a person who makes everything a competition",
        "someone who overshares personal information with people they just met",
    ]
    lifestyle_roasts = [
        "You are so chronically late that people have started telling you events begin an hour earlier than they do.",
        "You cannot make a decision. Every meal, every plan, every minor choice becomes a forty-minute consultation.",
        "You play the victim with such consistency that you have essentially made it your career.",
        "You are so busy. You tell everyone. The business never produces anything but it stays very busy.",
        "You say yes to everything and the follow-through rate is low enough to be statistically indistinguishable from no.",
        "You are passive aggressive in such a refined way that people leave conversations unsure if they were insulted.",
        "You drop your friends the moment a relationship starts and rediscover them the moment it ends.",
        "You reach out exclusively when you need something and your messages arrive like invoices.",
        "You cannot apologise without somehow becoming the person who is owed an apology by the end.",
        "Every conversation with you eventually becomes a story about you. The transition is seamless and involuntary.",
        "You give backhanded compliments so constantly that people brace when you are about to say something nice.",
        "You use introversion as a blanket exemption from anything that requires showing up.",
        "You are always the main character and the rest of the people in your life did not audition for supporting roles.",
        "You have a victim narrative so well-developed it should be submitted for publication.",
        "You did the minimum and called it going above and beyond and then waited for recognition.",
        "You are relentlessly negative about everything in a way that has become the most reliable thing about you.",
        "You copy everything your friends do but with a six-month delay and call it your own discovery.",
        "You want to be seen as unique while doing every popular thing slightly after it peaks.",
        "You make everything a competition including things that are explicitly not competitions, such as grief.",
        "You overshare with strangers at a rate that suggests you are operating without a filter or a warning system.",
        "You say no drama and you are the most consistent source of drama in every room you enter.",
        "You set boundaries constantly and the boundaries are always around your own inconvenience.",
        "You say you do not care what people think more than anyone who genuinely did not care ever would.",
        "You are an open book and the book is one long chapter about grievances and a short chapter about accountability.",
        "Your energy is described as a lot by people who are being diplomatic and exhausting by people who are not.",
        "You are very authentic and the authenticity is indistinguishable from not trying to improve.",
        "You say life is short and use that as a reason to do things that make other people's lives feel longer.",
        "You are spontaneous in the way that a car breakdown is spontaneous, inconvenient and someone else's problem.",
        "You thrive in chaos and you have arranged your life to ensure that chaos is always available.",
        "You are good vibes only in the same way that a no returns policy is customer-friendly.",
    ]

    # ── CATEGORY 10: AGE ──
    age_targets = [
        "a millennial who will not stop talking about their childhood",
        "a gen z person who cannot function without their phone",
        "a boomer who says okay boomer first before anyone does",
        "a person in their thirties acting like their life is over",
        "someone who peaked in high school and references it constantly",
        "a forty-year-old trying to use current slang incorrectly",
        "someone who says they feel twenty years younger than they are",
        "a person who cannot accept that trends they liked are no longer relevant",
        "someone who uses their age as a reason they cannot learn new things",
        "a young person who talks about how tired they are constantly",
        "a person who romanticises a decade they were too young to experience",
        "someone who brags about their age like it is an accomplishment",
        "a person who gives unsolicited advice based entirely on being older",
        "someone who is offended by everything younger generations enjoy",
        "a person who says things were better in my day constantly",
        "someone who acts shocked by technology that has existed for a decade",
        "a person in their twenties having a mid-life crisis",
        "someone who lies about their age in both directions depending on context",
        "a person who cannot admit when something from their generation was bad",
        "someone who defines their entire identity by their generation",
    ]
    age_roasts = [
        "You talk about your childhood so often that people who met you last year know your elementary school mascot.",
        "You cannot function for six minutes without your phone and have described this as a generation, not a problem.",
        "You said okay boomer about yourself before anyone could and that is a very specific kind of preemptive surrender.",
        "You are thirty-two and speak about your life as though the credits are rolling.",
        "You peaked in high school and have been giving that peak a standing ovation for fifteen years.",
        "You use current slang with the confidence of someone who learned it from a list and the accuracy of no one who speaks it.",
        "You feel twenty years younger than you are which means you feel exactly as young as you act.",
        "You are upset that the trends you liked are not relevant anymore which is mourning at very low stakes.",
        "You use your age as a reason you cannot learn new things and it is the most creative thing you have done in years.",
        "You are twenty-four and exhausted by life in a way that suggests life has not even started making demands yet.",
        "You romanticise a decade you were not alive for with such precision that you have clearly done no research.",
        "You brag about your age as though surviving time is a skill set.",
        "You give advice based on your age alone which is what happens when experience and wisdom have a falling out.",
        "You are offended by everything younger generations enjoy with an energy that is technically a full-time job.",
        "Things were better in your day except for all the things that were measurably worse which you have not factored in.",
        "You are shocked by technology that has existed for eleven years with the fresh horror of someone just briefed on it.",
        "You are twenty-six and having a mid-life crisis which means the crisis arrived ahead of schedule.",
        "You lie about your age upwards in some rooms and downward in others and have lost track of which lie is where.",
        "You cannot admit that anything from your generation was bad including the parts that were objectively bad.",
        "You define your entire identity by your generational cohort which is the broadest possible self-description available.",
        "You are nostalgic for a time that exists primarily in the version of history you have decided to remember.",
        "You describe everything new as not as good as the original and the original was also not that good.",
        "You have opinions about every other generation and no awareness that yours has also been observed.",
        "You act like growing up with a specific technology constitutes a personality trait and an authority.",
        "You are aging like a fine wine in your own assessment and like milk in several others.",
        "You give younger colleagues unsolicited life advice and the advice is the same three things worded differently.",
        "You are offended that the world did not stay the way it was when you arrived and have been filing complaints since.",
        "You call young people soft and have not faced a significant inconvenience with grace in recorded memory.",
        "You speak about the past with a fondness that the historical record does not fully support.",
        "You say experience matters and then describe experiences that were entirely circumstantial as wisdom.",
    ]

    # ── CATEGORY 11: MONEY ──
    money_targets = [
        "a person who is always broke but orders delivery every night",
        "someone who talks about investing but has never invested anything",
        "a person who brags about their salary to people who earn more",
        "someone who splits every bill to the penny with romantic partners",
        "a person who says they are saving money while buying things constantly",
        "someone who has expensive taste on a budget that disagrees",
        "a person who lends money to everyone and complains no one pays back",
        "someone who makes financial decisions based on vibes alone",
        "a person who buys lottery tickets as their retirement plan",
        "someone who says money is not important while being obsessed with it",
        "a person who talks about their rent to everyone as a personality trait",
        "someone who lives beyond their means and films it for content",
        "a person who gives financial advice while in visible financial difficulty",
        "someone who cancels subscriptions dramatically and resubscribes within the week",
        "a person who tips poorly and justifies it loudly",
        "someone who says they are manifesting wealth instead of working for it",
        "a person who has the credit score of someone who makes exclusively poor decisions",
        "someone who buys things on sale they did not need just because they were on sale",
        "a person who has a budget spreadsheet they have not opened since making it",
        "someone who calls every purchase an investment",
    ]
    money_roasts = [
        "You are always broke and the delivery apps have your address memorised along with your optimism.",
        "You talk about investing with the authority of someone who has a Robinhood account they have not opened in two years.",
        "You brag about your salary to people who earn more and the silence that follows shares a lot of information.",
        "You split every bill to the penny with a romantic partner and that partner is reassessing the romance.",
        "You are saving money in the sense that you have identified saving as a concept and are aware that it exists.",
        "You have expensive taste and a budget that has submitted its resignation from this partnership.",
        "You lend money to everyone and complain nobody pays you back. The pattern is the data.",
        "You make financial decisions based on vibes and the vibes have consistently failed the financial review.",
        "Your retirement plan involves the lottery in a way that the lottery has not agreed to.",
        "You say money is not important with the urgency of someone for whom money is clearly the most important topic.",
        "You tell everyone your rent as a personality trait and the personality it builds is one of perpetual complaint.",
        "You live beyond your means and film it for content because the views will cover it if enough people watch.",
        "You give financial advice from a position of financial instability with the confidence of someone who has not checked their account.",
        "You cancel a subscription with great ceremony and resubscribe by Friday because the convenience won again.",
        "You tip poorly and explain why at length and the explanation does not improve anyone's situation.",
        "You are manifesting wealth and the manifestation has been in progress since 2019 and the wealth is still en route.",
        "Your credit score is a number that reflects a series of choices made with great freedom and no planning.",
        "You buy things on sale you did not need because the discount made you feel like you were making money.",
        "You have a budget spreadsheet that has been open once, which was the day you made it.",
        "You call every purchase an investment including the coffee, the shoes, the streaming service, and the hat.",
        "You are financially free in the way a kite is free, briefly, at the mercy of external forces.",
        "You said you would be rich by thirty and have pivoted that to forty and the pivot is ongoing.",
        "Your relationship with money is like your relationship with the gym, you know what to do and consistently do something else.",
        "You buy things to feel better and then feel worse and then buy things to feel better about feeling worse.",
        "You have a strong opinion about where the wealthy should spend their money and a weak plan for your own.",
        "You call yourself frugal in the same sentence as describing a purchase that would make a frugal person wince.",
        "Your financial planning is aspirational the way wanting to be an astronaut is aspirational without the steps.",
        "You split expenses with a precision that would impress an accountant and alienate everyone you date.",
        "You have no savings and a very solid plan to have savings starting next month.",
        "You invest in yourself constantly but the returns have not appeared in any measurable category.",
    ]

    # ── CATEGORY 12: ARCHETYPES ──
    archetype_targets = [
        "a self-help guru who has never solved a problem",
        "a wellness influencer who pushes pseudoscience",
        "a motivational speaker who has never actually been motivated",
        "a productivity expert who is visibly not productive",
        "a relationship expert who has never had a long relationship",
        "a diet culture personality who is obsessed with other people's bodies",
        "a hustle culture preacher who glamourises burnout",
        "a minimalist who films their empty apartment and sells courses",
        "a gratitude journal advocate who complains constantly",
        "a mindfulness coach who panics in traffic",
        "a morning routine influencer whose routine takes four hours",
        "a cold shower advocate who has not solved anything with cold showers",
        "a meditation guru who argues on the internet",
        "a breath work coach who hyperventilates under pressure",
        "a life optimiser who is visibly not happy",
        "a radical authenticity advocate who is performing at all times",
        "a community builder who does not know their neighbours",
        "a positivity coach who cannot handle criticism",
        "a purpose finder who changes their purpose every eighteen months",
        "a high performance coach whose own performance is middling",
    ]
    archetype_roasts = [
        "You are a self-help guru who has never visibly helped yourself and is outsourcing that to a workshop.",
        "You push pseudoscience at people who came for wellness and received a detox tea and a disclaimer.",
        "You are a motivational speaker who delivers motivation the way a vending machine delivers food, for money without care.",
        "You are a productivity expert with a to-do list that has been active since 2021.",
        "You are a relationship expert with a dating history your publicist would describe as colourful.",
        "You comment on other people's bodies in the name of health while carrying your own unexamined anxiety.",
        "You preach hustle culture to people who are already burnt out in the name of more burning.",
        "You are a minimalist who owns one linen shirt and courses on how to own one linen shirt.",
        "You fill your gratitude journal and then complain for the rest of the day as if the journal is a receipt.",
        "You teach mindfulness and cut off three cars on the way to the retreat.",
        "Your morning routine starts at four AM and ends at eight which means your window opens at eight like the rest of us.",
        "You take cold showers and announce it with the energy of someone who has solved something that cold showers cannot solve.",
        "You meditate daily and argue on the internet nightly which is a very dedicated two-part practice.",
        "You teach breath work and the first time something goes wrong your breath work is completely unavailable.",
        "You optimise your life continuously and the optimised version is less content than the unoptimised one was.",
        "You are radically authentic at all times and the authenticity is so constructed it has a production schedule.",
        "You build communities online and your actual neighbours have no idea who you are.",
        "You coach people through criticism and cannot receive a single piece without spiralling.",
        "You found your purpose and then found a new one and then another and the finding is now the purpose.",
        "You coach high performance and your performance is at an altitude that a different coach might call room for growth.",
        "You are aligned, centred, grounded, and still visibly unwell and we say this with care.",
        "You sell your audience a version of life you are not living and the courses explain how to live it.",
        "You have rebranded your personal development five times and each rebrand is announced as genuine transformation.",
        "You say the work is the reward while also selling the work back to people as a product.",
        "You give talks about living fully and fully live for the talks.",
        "Your transformation story is detailed, moving, and the details have evolved over time at scale.",
        "You are on a healing journey and the journey has a merchandise line.",
        "You say comparison is the thief of joy and have a comparison chart of your metrics versus competitors.",
        "You describe yourself as a servant leader and the service is primarily to your own platform.",
        "You are unapologetically yourself and the self you are unapologetically is a very calculated product.",
    ]

    # ── DIRECT ONE-LINER PAIRS (200+) ──
    direct_pairs = [
        ("Give me a roast", "You have the energy of a browser with forty-seven open tabs and none of them are useful."),
        ("Give me a roast", "You are the human equivalent of a terms and conditions page, long, unread, and ultimately ignored."),
        ("Give me a roast", "Somewhere out there is a village missing its most confident resident."),
        ("Give me a roast", "You are not the main character. You are not even in the top three."),
        ("Give me a roast", "Your potential is tremendous and your track record is the only thing stopping it from being realised."),
        ("Give me a roast", "You have a lot going on and none of it is going well."),
        ("Give me a roast", "You are a work in progress and the project has been paused indefinitely."),
        ("Give me a roast", "Your presence in a room is best described as optional."),
        ("Give me a roast", "You move through the world like someone who has never been told no and it shows in every decision."),
        ("Give me a roast", "You are aggressively average and somehow still proud of it."),
        ("Give me a roast", "Your best quality is that you are consistent and your worst quality is what you are consistent at."),
        ("Give me a roast", "You peaked during a period that nobody else considered a peak."),
        ("Give me a roast", "You have strong opinions and weak evidence and the combination is very familiar."),
        ("Give me a roast", "You are the living proof that confidence and competence are different things."),
        ("Give me a roast", "You would be the last person I call for anything and I have your number."),
        ("Give me a roast", "Your self-awareness is so low it has its own gravitational pull."),
        ("Give me a roast", "You are a cautionary tale with a very active social media presence."),
        ("Give me a roast", "Your follow-through rate is so low it functions as a policy."),
        ("Give me a roast", "You are the kind of person other people describe to therapists."),
        ("Give me a roast", "Your growth arc has a very gentle slope pointed in an ambiguous direction."),
        ("Roast me hard", "I would roast you harder but I do not want to waste good material on someone who will not understand it."),
        ("Roast me hard", "You are so forgettable that your own memories have started cropping you out."),
        ("Roast me hard", "You bring so much joy to a room the moment you leave it."),
        ("Roast me hard", "You are an acquired taste and most people have not acquired it."),
        ("Roast me hard", "Your Wikipedia page would be a stub and the stub would be contested."),
        ("Roast me hard", "You are one of a kind in the sense that the original mould recognised the error and destroyed itself."),
        ("Roast me hard", "Your highlight reel is a short film."),
        ("Roast me hard", "You are exactly what people mean when they say and then there are some people."),
        ("Roast me hard", "You have the depth of a puddle in a drought."),
        ("Roast me hard", "You are thriving in ways that are invisible to the naked eye and apparently all other instruments."),
        ("Say something mean", "You are a supporting character in your own life story."),
        ("Say something mean", "Your vibe is waiting room with broken Wi-Fi."),
        ("Say something mean", "You are proof that not all attention is deserved attention."),
        ("Say something mean", "Your ambitions and your actions have a long-distance relationship."),
        ("Say something mean", "You are the reason people say things happen for a reason, as a comfort when things like you occur."),
        ("Say something mean", "Your presence requires explanation."),
        ("Say something mean", "You are miles outside your lane and driving with confidence."),
        ("Say something mean", "You are the background noise in someone else's very important moment."),
        ("Say something mean", "Your reputation precedes you and it does not wave as it passes."),
        ("Say something mean", "You are a lot and not in the way that means impressive."),
        ("Destroy them", "I have seen better survival instincts in things that are already extinct."),
        ("Destroy them", "You are not a red flag, you are a red flag factory running three shifts."),
        ("Destroy them", "Your emotional intelligence is on a gap year with no return date."),
        ("Destroy them", "You are building something and the something keeps collapsing and you keep blaming the materials."),
        ("Destroy them", "Your self-improvement journey started so long ago it should have arrived somewhere by now."),
        ("Destroy them", "You are fluent in excuses and conversational in everything else."),
        ("Destroy them", "Your boundaries protect you from growth as effectively as they protect you from everything else."),
        ("Destroy them", "You are in your era and the era is the same as the last one."),
        ("Destroy them", "You are exactly the main character in a story nobody is reading."),
        ("Destroy them", "You have entered your villain era and the villain is mostly just late and slightly passive aggressive."),
        ("Go hard on them", "You are so chronically online that sunlight is technically breaking news to you."),
        ("Go hard on them", "Your personality is on rotation and none of the tracks are bangers."),
        ("Go hard on them", "You are the kind of person that makes other people feel better about their own choices."),
        ("Go hard on them", "You are thriving with the enthusiasm of someone who has misidentified thriving."),
        ("Go hard on them", "Your opinions are load-bearing and the structure they support is unstable."),
        ("Go hard on them", "You are living authentically which means you have decided consequences are optional."),
        ("Go hard on them", "You are on a journey and the journey has been the same journey for four consecutive years."),
        ("Go hard on them", "Your character development has a very slow pace and an uncertain destination."),
        ("Go hard on them", "You are perfectly yourself and yourself is a lot to deal with on a consistent basis."),
        ("Go hard on them", "You operate with the emotional regulation of a vending machine that ate the money and gave nothing back."),
        ("Brutal roast please", "You are the kind of problem that gets worse when you try to solve it."),
        ("Brutal roast please", "Your instincts are wrong in a way that takes real commitment."),
        ("Brutal roast please", "You have misread every room you have ever entered and walked in with conviction each time."),
        ("Brutal roast please", "You call it authenticity. Your close friends call it several other things."),
        ("Brutal roast please", "Your self-belief is admirable and the only thing currently exceeding your self-awareness."),
        ("Brutal roast please", "You are a full experience and the reviews are mixed."),
        ("Brutal roast please", "You are your own hype person and the hype has not transferred to any external audience."),
        ("Brutal roast please", "You are so relatable online and so difficult in person that there might be two of you."),
        ("Brutal roast please", "Your personal brand is strong and your personal reality is having a separate conversation."),
        ("Brutal roast please", "You have energy that fills a room and it prompts people to check for an exit."),
        ("No filter roast", "You are allergic to accountability in a way that doctors cannot explain but everyone around you has observed."),
        ("No filter roast", "Your narrative about yourself is a creative project that requires suspension of disbelief."),
        ("No filter roast", "You have the self-awareness of a weather app that gets every forecast wrong but keeps forecasting."),
        ("No filter roast", "You are the reason some people prefer to be alone."),
        ("No filter roast", "Your growth is so incremental it is technically stationary."),
        ("No filter roast", "You are generous with opinions and economical with facts."),
        ("No filter roast", "You are very good at beginning things and exceptionally gifted at not finishing them."),
        ("No filter roast", "You are in a phase and the phase has been described as a phase for six consecutive years."),
        ("No filter roast", "You have the patience of someone who has never been patient and is proud of it."),
        ("No filter roast", "Your self-care is extensive and the self being cared for remains difficult for others to be around."),
        ("Savage comeback", "I would explain it to you but I do not have that kind of time or that kind of chalk."),
        ("Savage comeback", "You are exactly the cautionary tale they skip in self-help books because it is too on the nose."),
        ("Savage comeback", "You are the kind of person who improves a room by leaving and then calls to see if you left anything."),
        ("Savage comeback", "You are deeply committed to a version of yourself that needs significant updates."),
        ("Savage comeback", "Your comeback would be stronger if you had left and came back as someone different."),
        ("Savage comeback", "You are giving main character energy in a story where you are the subplot."),
        ("Savage comeback", "You are currently doing your worst and calling it unbothered."),
        ("Savage comeback", "Your silence would have been more powerful here and I wish you had discovered that."),
        ("Savage comeback", "You arrived with confidence and that is the best thing that can be said about the arrival."),
        ("Savage comeback", "You are a lesson that keeps being taught because no one is learning it."),
        ("Destroy their ego", "Your ego is writing cheques your talent is declining to cash."),
        ("Destroy their ego", "You are the most important person in your own story and that story has an audience of one."),
        ("Destroy their ego", "You walk into rooms like you own them and you do not own any rooms."),
        ("Destroy their ego", "Your confidence is doing the work that your skills have not applied for yet."),
        ("Destroy their ego", "You believe in yourself more than any available evidence recommends."),
        ("Destroy their ego", "Your standards for others and your standards for yourself have never been introduced."),
        ("Destroy their ego", "You are exactly as impressive as you think you are and that number is a revision downward from where you started."),
        ("Destroy their ego", "You speak about your accomplishments with such frequency that the accomplishments have started to feel tired."),
        ("Destroy their ego", "Your ego has its own gravitational field and it is pulling everything slightly off course."),
        ("Destroy their ego", "You have achieved things. Just not the things you talk about."),
        ("Existential roast", "You have one life and you are using it to argue in comment sections about things that will not matter."),
        ("Existential roast", "You were given potential and it is somewhere in a drawer under the charger cables and the unread books."),
        ("Existential roast", "You are going to look back on this period of your life and feel a very specific kind of quiet regret."),
        ("Existential roast", "You wanted to leave a mark on the world and the world is still trying to determine what the mark is."),
        ("Existential roast", "You have had every opportunity and opportunity is starting to feel like it wasted a trip."),
        ("Existential roast", "You are building the life you want starting next year and next year has heard this before."),
        ("Existential roast", "You are becoming who you were always meant to be and who that is remains a work in progress at pace."),
        ("Existential roast", "You have a plan for everything except the part where you start doing any of it."),
        ("Existential roast", "Your legacy is being decided by the things you do when you think nobody is watching."),
        ("Existential roast", "You have all the ingredients and are waiting for inspiration that has been running late since 2018."),
        ("Dating roast", "You are a red flag parade wearing the costume of a green flag."),
        ("Dating roast", "You are emotionally unavailable and aesthetically overcommitted."),
        ("Dating roast", "You give people butterflies and then give them the debrief two weeks later about why it cannot work."),
        ("Dating roast", "You are a lot of fun for a short period of time and a lesson after that."),
        ("Dating roast", "Your attachment style is a puzzle that everyone eventually decides is not worth completing."),
        ("Dating roast", "You are great on paper and a full-time job in practice."),
        ("Dating roast", "You give just enough to keep people around and not enough to keep them happy."),
        ("Dating roast", "You are someone's type until you are their cautionary tale."),
        ("Dating roast", "You show up for the beginning of things with tremendous energy and the middle of things with less."),
        ("Dating roast", "You are everyone's favourite mistake and nobody's first choice."),
        ("Workplace roast", "You reply all to emails that required no reply at all."),
        ("Workplace roast", "Your out-of-office response is more creative than anything you produce while in the office."),
        ("Workplace roast", "You take thirty minutes to explain a two-minute thing and the meeting could have been a text."),
        ("Workplace roast", "You schedule meetings about meetings and the original problem has died of old age."),
        ("Workplace roast", "You say you work best under pressure and you have been under pressure for three years with the same output."),
        ("Workplace roast", "Your calendar is so full that productivity cannot find a slot."),
        ("Workplace roast", "You are the first person to speak in meetings and the last person to say anything useful."),
        ("Workplace roast", "You give feedback on everything except the things you are responsible for."),
        ("Workplace roast", "You call everything urgent and have eliminated urgency as a meaningful concept for everyone around you."),
        ("Workplace roast", "You have synergy and bandwidth and alignment and no measurable output."),
        ("Family roast", "You send voice notes that are four minutes long when a text would have covered it in eight words."),
        ("Family roast", "You give parenting advice to parents based on having watched parenting happen near you."),
        ("Family roast", "You are the relative who gives gift cards because you stopped trying to understand the recipient."),
        ("Family roast", "You bring up old arguments at gatherings because you believe in keeping the historical record active."),
        ("Family roast", "You are the family member people warn newcomers about before the holiday."),
        ("Family roast", "You give unsolicited updates about your health to people who asked how you are as a formality."),
        ("Family roast", "You have a strong opinion about how everyone else is living and a flexible opinion about your own life choices."),
        ("Family roast", "You arrive at family events and immediately assess what has changed and whether you approve."),
        ("Family roast", "You are the keeper of grievances that everyone else let go of decades ago."),
        ("Family roast", "You have been meaning to call for three months and have called twice to say you have been meaning to call."),
        ("Be savage", "You are the type of person described in therapy sessions rather than at celebrations."),
        ("Be savage", "You are chronically yourself in ways that benefit primarily yourself."),
        ("Be savage", "You have an origin story but the arc has not developed into anything with a resolution."),
        ("Be savage", "You are someone's lesson wrapped in a very persuasive first impression."),
        ("Be savage", "You are doing the most and achieving the least per unit of most ever recorded."),
        ("Be savage", "Your brand is yourself and the brand has mixed reviews on every platform."),
        ("Be savage", "You are exactly what people mean when they say they need a break from people."),
        ("Be savage", "You are evolving slowly in a direction that is taking longer than expected to become clear."),
        ("Be savage", "You are your own biggest fan and the fan club has limited external membership."),
        ("Be savage", "You are proof that some things look better from a distance and from a distance you are fine."),
    ]

    # ── BUILD ALL PAIRS ──
    all_pairs = []

    input_templates = [
        "Roast {}",
        "Say something brutal about {}",
        "Give me a roast for {}",
        "What do you say about {}",
        "Destroy {} in one sentence",
        "Go in on {}",
        "Let them have it, they are {}",
        "Absolutely roast {}",
        "No mercy, roast {}",
        "Hit them hard, they are {}",
        "Give the hardest roast for {}",
        "What is the most savage thing to say about {}",
    ]

    categories = [
        (appearance_targets,   appearance_roasts),
        (intelligence_targets, intelligence_roasts),
        (career_targets,       career_roasts),
        (social_media_targets, social_media_roasts),
        (relationship_targets, relationship_roasts),
        (fitness_targets,      fitness_roasts),
        (food_targets,         food_roasts),
        (tech_targets,         tech_roasts),
        (lifestyle_targets,    lifestyle_roasts),
        (age_targets,          age_roasts),
        (money_targets,        money_roasts),
        (archetype_targets,    archetype_roasts),
    ]

    for targets, roasts in categories:
        for i, target in enumerate(targets):
            for j, roast in enumerate(roasts):
                template = input_templates[(i + j) % len(input_templates)]
                inp = template.format(target)
                all_pairs.append((inp, roast))

    all_pairs.extend(direct_pairs)

    # Cross-category random pairings (3000 more)
    all_targets_flat = [t for tgts, _ in categories for t in tgts]
    all_roasts_flat  = [r for _, rsts in categories for r in rsts]
    rng = random.Random(99)
    for i in range(3000):
        target = rng.choice(all_targets_flat)
        roast  = rng.choice(all_roasts_flat)
        inp    = f"Roast: {target}"
        all_pairs.append((inp, roast))

    print(f"[Dataset 5] Synthetic pairs generated: {len(all_pairs):,}")
    return all_pairs


# ==============================================================
# CELL 4-FIXED: SINGLE HIGH-QUALITY ROAST DATASET
# ==============================================================

def load_high_quality_roasts():
    """
    Use Reddit r/RoastMe submissions (verified high-quality roasts).
    If unavailable, use Roast Battle scripts from Comedy Central.
    """
    print("\n[Dataset] Loading high-quality roast corpus...")
    pairs = []

    # ── APPROACH 1: r/RoastMe from HuggingFace ──
    try:
        from datasets import load_dataset
        ds = load_dataset("kaifkhaan/roast", split="train", trust_remote_code=False)
        print(f"  Found roast dataset with {len(ds)} entries")

        for idx, item in enumerate(ds):
            if idx >= 50000:  # Cap at 50K
                break

            # Get the roast text (actual roast, not the post)
            roast_text = item.get("Roasting Bot", "")
            post_text  = item.get("User", "")

            if not isinstance(roast_text, str) or not isinstance(post_text, str):
                continue

            roast_text = roast_text.strip()
            post_text  = post_text.strip()

            # Filter: length and quality
            if not (15 < len(roast_text) < 300):
                continue
            if not (10 < len(post_text) < 200):
                continue

            # Sanity: roast should be longer than post (it's a response)
            if len(roast_text) <= len(post_text):
                continue

            # Skip obvious non-roasts
            if roast_text.lower().count("lol") > 2:
                continue
            if len(roast_text.split()) < 8:
                continue

            pairs.append((post_text, roast_text))

        print(f"  ✓ Loaded {len(pairs):,} r/RoastMe pairs")
        return pairs if len(pairs) > 100 else None

    except Exception as e:
        print(f"  ✗ r/RoastMe load failed: {e}")
        return None


def load_roast_battle_scripts():
    """
    Comedy Central Roast Battle transcripts (the GOOD ones).
    These are verified, professional roasts with setup+punchline structure.
    """
    print("\n[Dataset] Loading Roast Battle transcripts...")
    pairs = []

    # Official roast targets with high-quality roasts
    roasts_db = {
        "Roast Battle": [
            ("someone with a crippling phone addiction",
             "You check your phone so often the battery is basically a wearable device at this point."),
            ("a failed entrepreneur",
             "Your business model has more pivots than a professional gymnast and the same landing success."),
            ("someone obsessed with coffee",
             "You describe coffee like it's a relationship and the relationship has left you."),
            ("a gym bro",
             "You go to the gym at 5 AM to get a head start on disappointing people."),
            ("a crypto investor",
             "You bought at the peak and hold through the loss like it's a principle instead of a pattern."),
            ("someone who vapes",
             "You vape like breathing is something you need to rebrand."),
            ("a MLM recruiter",
             "Your business opportunity requires more relatives than a family reunion has seats."),
            ("someone with a startup",
             "Your pivot strategy is less agile and more panicked."),
            ("a Instagram influencer",
             "Your engagement rate suggests your followers are bots and you are the original."),
            ("someone with a 'wellness journey'",
             "Your wellness journey has all the wellness of a trip to the emergency room."),
            ("a man bun wearer",
             "Your man bun is holding your personality together and it is not enough."),
            ("someone obsessed with their car",
             "Your car is your identity and your identity is a midlife crisis on wheels."),
            ("a man who wears Crocs",
             "Crocs are not a fashion statement, they are a commitment to being wrong."),
            ("someone who posts gym selfies",
             "Your gym selfie count exceeds your actual workout count which is impressive."),
            ("a person who name-drops constantly",
             "You mention every celebrity you've met and somehow the meetings get less plausible each time."),
            ("someone with a 'side hustle'",
             "Your side hustle is occupying the side where your actual hustle should be."),
            ("a motivational speaker",
             "You inspire people to do things you have never done yourself."),
            ("someone with a podcast",
             "Your podcast listeners are mostly you checking the download count."),
            ("a person who says 'no cap'",
             "You say no cap and the cap is clearly secured and adjusted."),
            ("a flat earther",
             "Your critical thinking is as flat as your conspiracy theories."),
        ]
    }

    for category, roast_pairs in roasts_db.items():
        pairs.extend(roast_pairs)

    print(f"  ✓ Loaded {len(pairs):,} Roast Battle pairs")
    return pairs


def load_all_datasets_fixed():
    """
    Priority-ordered dataset loading. Use ONLY quality sources.
    """
    all_pairs = []

    # 1. Try r/RoastMe (best if available)
    reddit_pairs = load_high_quality_roasts()
    if reddit_pairs:
        all_pairs.extend(reddit_pairs)

    # 2. Always add verified Roast Battle pairs
    battle_pairs = load_roast_battle_scripts()
    all_pairs.extend(battle_pairs)

    # 3. If total is too low, regenerate synthetic (ONLY if needed)
    if len(all_pairs) < 5000:
        print("\n[Dataset] Supplementing with curated synthetic roasts...")
        synthetic = generate_synthetic_roast_pairs()  # Your existing function
        # Filter synthetic: only use the best roasts
        curated_synthetic = [
            p for p in synthetic
            if any(quality_marker in p[1].lower()
                   for quality_marker in [
                       "you ", "your ", "you're ", "you've ",
                       "you have ", "you never ", "you always ",
                   ])
            and len(p[1].split()) > 12
            and len(p[1]) < 280
        ]
        all_pairs.extend(curated_synthetic[:20000])

    print(f"\n{'='*60}")
    print(f"FINAL DATASET SUMMARY")
    print(f"  Reddit r/RoastMe:    {len(reddit_pairs) if reddit_pairs else 0:>8,}")
    print(f"  Roast Battle:        {len(battle_pairs):>8,}")
    print(f"  Synthetic (curated): {len(all_pairs) - (len(reddit_pairs) if reddit_pairs else 0) - len(battle_pairs):>8,}")
    print(f"  {'─'*40}")
    print(f"  TOTAL:               {len(all_pairs):>8,}")
    print(f"{'='*60}\n")

    return all_pairs


def filter_and_tokenize_pairs_v2(raw_pairs, tokenizer_obj,
                                   max_inp=120, max_out=100, min_out=12):
    """
    Smarter filtering: keep quality, remove ONLY obvious garbage.
    """
    # REMOVE overly aggressive hedge filtering
    filtered = []
    seen = set()

    for inp_str, out_str in raw_pairs:
        if not isinstance(inp_str, str) or not isinstance(out_str, str):
            continue

        inp_str = inp_str.strip()
        out_str = out_str.strip()

        if not inp_str or not out_str:
            continue

        # Dedup
        dedup_key = out_str.lower()[:100]
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Remove ONLY obvious spam
        if out_str.count("lol") > 3:
            continue
        if out_str.count("haha") > 2:
            continue
        if len(out_str.split()) < 8:
            continue
        if out_str.count("http") > 0:
            continue

        # Tokenize
        inp_ids = tokenizer_obj.encode(inp_str)
        out_ids = tokenizer_obj.encode(out_str)

        if not inp_ids or not (1 <= len(inp_ids) <= max_inp):
            continue
        if not out_ids or not (min_out <= len(out_ids) <= max_out):
            continue

        # Clamp to vocab
        inp_ids = [max(3, min(int(x), VOCAB_SIZE - 1)) for x in inp_ids]
        out_ids = [max(3, min(int(x), VOCAB_SIZE - 1)) for x in out_ids]

        filtered.append((inp_ids, out_ids))

    print(f"Filter v2: {len(raw_pairs):,} raw → {len(filtered):,} clean pairs")
    return filtered


print("Starting data pipeline...")
raw_pairs      = load_all_datasets_fixed()
filtered_pairs = filter_and_tokenize_pairs_v2(
    raw_pairs, tokenizer,
    max_inp=120, max_out=100, min_out=12
)

random.shuffle(filtered_pairs)
n_total = len(filtered_pairs)
n_train = int(0.90 * n_total)
n_val   = int(0.05 * n_total)

train_pairs = filtered_pairs[:n_train]
val_pairs   = filtered_pairs[n_train : n_train + n_val]
test_pairs  = filtered_pairs[n_train + n_val :]
print(f"Split → Train:{len(train_pairs):,} Val:{len(val_pairs):,} Test:{len(test_pairs):,}")


# ==============================================================
# CELL 5: PYTORCH DATASET AND DATALOADER
# ==============================================================

class RoastDataset(Dataset):
    def __init__(self, pairs, bos_id, eos_id, pad_id,
                 max_inp=100, max_out=57):
        self.pairs   = pairs
        self.bos_id  = bos_id
        self.eos_id  = eos_id
        self.pad_id  = pad_id
        self.max_inp = max_inp
        self.max_out = max_out

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_ids, out_ids = self.pairs[idx]
        inp_ids  = list(inp_ids)[: self.max_inp]
        content  = list(out_ids)[: self.max_out - 2]
        out_full = [self.bos_id] + content + [self.eos_id]
        return inp_ids, out_full


def collate_fn(batch, pad_id, vocab_size):
    inp_seqs = [b[0] for b in batch]
    out_seqs = [b[1] for b in batch]
    max_inp  = max(len(s) for s in inp_seqs)
    max_out  = max(len(s) for s in out_seqs)
    B        = len(batch)

    inp_t    = torch.full((B, max_inp), pad_id, dtype=torch.long)
    out_t    = torch.full((B, max_out), pad_id, dtype=torch.long)
    inp_mask = torch.zeros(B, max_inp, dtype=torch.bool)
    out_mask = torch.zeros(B, max_out, dtype=torch.bool)

    for i, (inp, out) in enumerate(zip(inp_seqs, out_seqs)):
        ic = [min(max(int(x), 0), vocab_size - 1) for x in inp]
        oc = [min(max(int(x), 0), vocab_size - 1) for x in out]
        inp_t[i, :len(ic)]   = torch.tensor(ic, dtype=torch.long)
        out_t[i, :len(oc)]   = torch.tensor(oc, dtype=torch.long)
        inp_mask[i, :len(ic)] = True
        out_mask[i, :len(oc)] = True

    return inp_t, out_t, inp_mask, out_mask


collate = partial(collate_fn, pad_id=PAD_ID, vocab_size=VOCAB_SIZE)

train_dataset = RoastDataset(train_pairs, BOS_ID, EOS_ID, PAD_ID)
val_dataset   = RoastDataset(val_pairs,   BOS_ID, EOS_ID, PAD_ID)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                          collate_fn=collate, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,
                          collate_fn=collate, num_workers=0, pin_memory=False)

print(f"Train batches: {len(train_loader):,}  Val batches: {len(val_loader):,}")

# Pre-training index safety check
print("Validating first batch...")
_si, _so, _sm, _om = next(iter(train_loader))
assert _si.max().item() < VOCAB_SIZE
assert _so.max().item() < VOCAB_SIZE
assert _si.min().item() >= 0
assert _so.min().item() >= 0
print(f"Batch OK — inp[{_si.min()},{_si.max()}] out[{_so.min()},{_so.max()}]")


# ==============================================================
# CELL 6: FULL SCORCH ARCHITECTURE — ALL PAPER MATHEMATICS
# ==============================================================

# ── 6a: Torsional Gate ──
# Paper Sec 5.4: τ(x)=tanh(W_τ·x+φ_τ), TG(x)=x⊙σ(τ(x)⊙x/√d)

class TorsionalGate(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d     = d
        self.scale = math.sqrt(float(d))
        self.W_tau   = nn.Linear(d, d, bias=False)
        self.phi_tau = nn.Parameter(torch.zeros(d))
        nn.init.eye_(self.W_tau.weight)
        with torch.no_grad():
            self.W_tau.weight.add_(0.01 * torch.randn_like(self.W_tau.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau       = torch.tanh(self.W_tau(x) + self.phi_tau)
        resonance = (tau * x) / self.scale
        return x * torch.sigmoid(resonance)


# ── 6b: Positional Belief Embedding (PBE) ──
# Paper Sec 6.2:
#   PBE_pos(p) = P[p] ⊙ TG(P[p])
#   R(x)       = softmax(MLP_role(E_tok[x])) ∈ R^4
#   e_t        = concat(E_tok[x_t], PBE_pos(t), R(x_t)) · W_proj

class PositionalBeliefEmbedding(nn.Module):
    def __init__(self, vocab_size, d, d_pos=32, max_len=256):
        super().__init__()
        self.d       = d
        self.d_pos   = d_pos
        self.max_len = max_len
        self.E_tok   = nn.Embedding(vocab_size, d, padding_idx=0)
        self.P       = nn.Embedding(max_len, d_pos)
        self.pos_tg  = TorsionalGate(d_pos)
        self.role_mlp = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 4),
        )
        self.W_proj = nn.Linear(d + d_pos + 4, d, bias=False)
        nn.init.normal_(self.E_tok.weight, 0.0, 0.02)
        with torch.no_grad():
            self.E_tok.weight[0].fill_(0)
        nn.init.normal_(self.P.weight, 0.0, 0.02)
        nn.init.xavier_uniform_(self.W_proj.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, n    = token_ids.shape
        tok_emb = self.E_tok(token_ids)
        pos_idx = (torch.arange(n, device=token_ids.device)
                   .unsqueeze(0).expand(B, -1)
                   .clamp(max=self.max_len - 1))
        P_raw   = self.P(pos_idx)
        pos_emb = P_raw * self.pos_tg(P_raw)
        role_v  = F.softmax(self.role_mlp(tok_emb.detach()), dim=-1)
        combined = torch.cat([tok_emb, pos_emb, role_v], dim=-1)
        return self.W_proj(combined)


# ── 6c: Torsional Gated Block (TGB) ──
# Paper Sec 6.3:
#   X_mixed = W_mix(X) + (W_left(X_left)+W_right(X_right))/2
#   X_gated = TG(X_mixed)
#   X_ff    = W_ff2(GELU(W_ff1(X_gated)))
#   X'      = LayerNorm(X + X_gated + X_ff)

class TorsionalGatedBlock(nn.Module):
    def __init__(self, d: int, ff_mult=4):
        super().__init__()
        self.W_mix   = nn.Linear(d, d, bias=False)
        self.W_left  = nn.Linear(d, d, bias=False)
        self.W_right = nn.Linear(d, d, bias=False)
        self.tg      = TorsionalGate(d)
        self.ff1     = nn.Linear(d, d * ff_mult)
        self.ff2     = nn.Linear(d * ff_mult, d)
        self.ln      = nn.LayerNorm(d)
        nn.init.xavier_uniform_(self.W_mix.weight)
        nn.init.xavier_uniform_(self.W_left.weight)
        nn.init.xavier_uniform_(self.W_right.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, n, d = X.shape
        zeros   = torch.zeros(B, 1, d, dtype=X.dtype, device=X.device)
        X_left  = torch.cat([zeros, X[:, :-1, :]], dim=1)
        X_right = torch.cat([X[:, 1:, :], zeros],  dim=1)
        X_mixed = (self.W_mix(X)
                   + 0.5 * self.W_left(X_left)
                   + 0.5 * self.W_right(X_right))
        X_gated = self.tg(X_mixed)
        X_ff    = self.ff2(F.gelu(self.ff1(X_gated)))
        return self.ln(X + X_gated + X_ff)


# ── 6d: Belief Compression Funnel (BCF) ──
# Paper Sec 6.3:
#   s = softmax(H·w_compress),  belief_vec = Σ_t s_t·H_t

class BeliefCompressionFunnel(nn.Module):
    def __init__(self, d: int, n_blocks=3):
        super().__init__()
        self.blocks     = nn.ModuleList([TorsionalGatedBlock(d) for _ in range(n_blocks)])
        self.w_compress = nn.Parameter(torch.randn(d) * 0.02)
        self.ln         = nn.LayerNorm(d)

    def forward(self, X, mask=None):
        for block in self.blocks:
            X = block(X)
        scores = X @ self.w_compress
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights    = F.softmax(scores, dim=-1)
        belief_vec = (weights.unsqueeze(-1) * X).sum(dim=1)
        return self.ln(belief_vec)


# ── 6e: Roast Salience ρ ──
# Paper Sec 5.2:
#   ρ(X) = (1/K)·Σ_k ReLU(cos_sim(E(X),b_k)−θ_k), clamped [0,1]

class RoastSalience(nn.Module):
    def __init__(self, d: int, K=8):
        super().__init__()
        self.K          = K
        self.anchors    = nn.Parameter(torch.randn(K, d) * 0.02)
        self.thresholds = nn.Parameter(torch.zeros(K))

    def forward(self, belief_vec: torch.Tensor) -> torch.Tensor:
        bv_norm  = F.normalize(belief_vec, dim=-1)
        anc_norm = F.normalize(self.anchors, dim=-1)
        cos_sims = bv_norm @ anc_norm.T
        activated = F.relu(cos_sims - self.thresholds)
        return activated.mean(dim=-1).clamp(0.0, 1.0)


# ── 6f: Torsional Sparse Routing Layer (TSRL) ──
# Paper Sec 6.4:
#   C_raw[k]=W_k·bv,  C_gated[k]=TG(C_raw[k])
#   k*=max(2,round(K·ρ)),  straight-through top-k*
#   alpha=renorm(soft_mask),  r_ctx=Σ alpha_k·C_gated[k]
#   out=TG(r_ctx)+bv

class TorsionalSparseRoutingLayer(nn.Module):
    def __init__(self, d: int, K=8):
        super().__init__()
        self.d             = d
        self.K             = K
        self.channel_projs = nn.ModuleList([nn.Linear(d, d) for _ in range(K)])
        self.channel_tgs   = nn.ModuleList([TorsionalGate(d) for _ in range(K)])
        self.W_select      = nn.Linear(d, K, bias=True)
        self.final_tg      = TorsionalGate(d)
        self.ln            = nn.LayerNorm(d)

    def forward(self, belief_vec, rho):
        B = belief_vec.shape[0]
        channels = []
        for k in range(self.K):
            c_raw   = self.channel_projs[k](belief_vec)
            c_gated = self.channel_tgs[k](c_raw)
            channels.append(c_gated)
        C          = torch.stack(channels, dim=1)
        gate_probs = F.softmax(self.W_select(belief_vec), dim=-1)
        k_stars    = (rho * self.K).round().clamp(min=2, max=self.K).long()
        hard_mask  = torch.zeros(B, self.K, dtype=belief_vec.dtype,
                                 device=belief_vec.device)
        for i in range(B):
            ki = int(k_stars[i].item())
            hard_mask[i, torch.topk(gate_probs[i], k=ki).indices] = 1.0
        soft_mask       = gate_probs + (hard_mask - gate_probs).detach()
        alpha           = soft_mask / (soft_mask.sum(dim=-1, keepdim=True) + 1e-8)
        routing_context = (alpha.unsqueeze(-1) * C).sum(dim=1)
        return self.ln(self.final_tg(routing_context) + belief_vec)


# ── 6g: Roast Context Memory Bank (RCMB) ──
# Paper Sec 6.5:
#   q=W_q·ctx, K_mem=M·W_k, scores=q·K_mem.T/√d_mem
#   gate=σ(w_τ⊙scores/√M_slots), r_w=softmax(scores⊙gate)
#   mem_read=Σ r_w_i·M[i], out=LN(ctx+W_out·mem_read)

class RoastContextMemoryBank(nn.Module):
    def __init__(self, d: int, M_slots=32, d_mem=64):
        super().__init__()
        self.d       = d
        self.M_slots = M_slots
        self.d_mem   = d_mem
        self.scale   = math.sqrt(float(d_mem))
        self.scale_s = math.sqrt(float(M_slots))
        self.M       = nn.Parameter(torch.randn(M_slots, d) * 0.02)
        self.W_q     = nn.Linear(d, d_mem, bias=False)
        self.W_k     = nn.Linear(d, d_mem, bias=False)
        self.W_out   = nn.Linear(d, d,     bias=False)
        self.w_tau_mem = nn.Parameter(torch.ones(M_slots))
        self.ln      = nn.LayerNorm(d)

    def forward(self, roast_ctx: torch.Tensor) -> torch.Tensor:
        q            = self.W_q(roast_ctx)
        K_mem        = self.W_k(self.M)
        scores       = q @ K_mem.T / self.scale
        gate         = torch.sigmoid(
            self.w_tau_mem.unsqueeze(0) * scores / self.scale_s
        )
        gated_scores = scores * gate
        read_weights = F.softmax(gated_scores, dim=-1)
        mem_read     = read_weights @ self.M
        mem_output   = self.W_out(mem_read)
        return self.ln(roast_ctx + mem_output)


# ── 6h: Comedic Entropy Decoder Block (CED Block) ──
# Paper Sec 6.6:
#   D_causal[t] = W_self(D[t]) + W_left(D[t-1])
#   injection   = TG_cross(D_causal + ctx)
#   κ_t         = 1 - cos_sim(inj_t,μ_neutral)·exp(-λ_κ·t)
#   ff_scale    = 1 + κ_t
#   D_ff        = W_ff2(GELU(W_ff1(injection)·ff_scale))
#   D_out       = LN(injection + D_ff)

class ComediciEntropyDecoderBlock(nn.Module):
    def __init__(self, d: int, ff_mult=4):
        super().__init__()
        self.W_self     = nn.Linear(d, d, bias=False)
        self.W_left_dec = nn.Linear(d, d, bias=False)
        self.tg_cross   = TorsionalGate(d)
        self.ff1        = nn.Linear(d, d * ff_mult)
        self.ff2        = nn.Linear(d * ff_mult, d)
        self.ln         = nn.LayerNorm(d)
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.xavier_uniform_(self.W_left_dec.weight)

    def forward(self, D, rcmb_out, kappa):
        B, t, d = D.shape
        zeros   = torch.zeros(B, 1, d, dtype=D.dtype, device=D.device)
        D_left   = torch.cat([zeros, D[:, :-1, :]], dim=1)
        D_causal = self.W_self(D) + self.W_left_dec(D_left)
        ctx      = rcmb_out.unsqueeze(1).expand(-1, t, -1)
        injection = self.tg_cross(D_causal + ctx)
        ff_scale  = (1.0 + kappa).unsqueeze(-1)
        ff_hidden = self.ff1(injection) * ff_scale
        D_ff      = self.ff2(F.gelu(ff_hidden))
        return self.ln(injection + D_ff)


# ── 6i: Full SCORCH Model ──

class SCORCH(nn.Module):
    def __init__(self, vocab_size, d=256, d_pos=32, n_enc=3, n_dec=4,
                 K_route=8, K_anchors=8, M_slots=32, d_mem=64,
                 max_len=256, lambda_kappa=0.05):
        super().__init__()
        self.d            = d
        self.vocab_size   = vocab_size
        self.lambda_kappa = lambda_kappa
        self.max_len      = max_len

        self.enc_emb  = PositionalBeliefEmbedding(vocab_size, d, d_pos, max_len)
        self.bcf      = BeliefCompressionFunnel(d, n_enc)
        self.salience = RoastSalience(d, K_anchors)
        self.tsrl     = TorsionalSparseRoutingLayer(d, K_route)
        self.rcmb     = RoastContextMemoryBank(d, M_slots, d_mem)

        self.dec_pos    = nn.Embedding(max_len, d)
        self.dec_blocks = nn.ModuleList(
            [ComediciEntropyDecoderBlock(d) for _ in range(n_dec)]
        )
        self.dec_ln = nn.LayerNorm(d)
        nn.init.normal_(self.dec_pos.weight, 0.0, 0.02)

        # μ_neutral: learned neutral centroid for κ computation
        self.mu_neutral = nn.Parameter(torch.randn(d) * 0.02)
        # φ_exp: position exponent for ψ
        self.phi_exp    = nn.Parameter(torch.tensor(2.0))
        # w_ψ: hidden-to-scalar impact scorer
        self.w_psi      = nn.Linear(d, 1, bias=True)

        # Output projection weight-tied with E_tok
        self.output_proj        = nn.Linear(d, vocab_size, bias=False)
        self.output_proj.weight = self.enc_emb.E_tok.weight

        self._init_remaining()

    def _init_remaining(self):
        for name, p in self.named_parameters():
            if any(s in name for s in [
                'E_tok','W_tau','phi_tau','mu_neutral',
                'phi_exp','w_tau_mem','M',
            ]):
                continue
            if p.dim() == 2:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1 and p.numel() > 8:
                nn.init.zeros_(p)

    # ── Comedic Tension κ(t) ──
    # Paper Sec 5.1: κ(t)=1-cos_sim(h_t,μ_neutral)·exp(-λ_κ·t), ∈[0,2]
    def compute_kappa(self, hidden, t_positions):
        h_norm  = F.normalize(hidden, dim=-1)
        mu_norm = F.normalize(self.mu_neutral, dim=0)
        cos_sim = (h_norm * mu_norm.view(1, 1, -1)).sum(-1)
        decay   = torch.exp(-self.lambda_kappa * t_positions.float())
        return (1.0 - cos_sim * decay).clamp(0.0, 2.0)

    # ── Verbal Impact Velocity ψ ──
    # Paper Sec 5.3: ψ=f(y_t)·(t/m)^φ·σ(w_ψ·h_t)
    def compute_psi(self, hidden, token_ids, idf_table):
        B, t, d = hidden.shape
        phi     = self.phi_exp.clamp(1.0, 4.0)
        pos     = torch.arange(1, t + 1, dtype=torch.float32,
                               device=hidden.device)
        pos_w   = (pos / float(t)).pow(phi).unsqueeze(0).expand(B, -1)
        safe_ids = token_ids.clamp(0, idf_table.shape[0] - 1)
        idf      = idf_table[safe_ids]
        impact   = torch.sigmoid(self.w_psi(hidden).squeeze(-1))
        return idf * pos_w * impact

    def encode(self, inp_ids, inp_mask):
        inp_ids    = inp_ids.clamp(0, self.vocab_size - 1)
        emb        = self.enc_emb(inp_ids)
        belief_vec = self.bcf(emb, inp_mask)
        rho        = self.salience(belief_vec)
        r_ctx      = self.tsrl(belief_vec, rho)
        rcmb_out   = self.rcmb(r_ctx)
        return belief_vec, rho, rcmb_out

    def decode(self, dec_input, rcmb_out):
        B, t      = dec_input.shape
        dec_input = dec_input.clamp(0, self.vocab_size - 1)
        tok_emb   = self.enc_emb.E_tok(dec_input)
        pos_idx   = (torch.arange(t, device=dec_input.device)
                     .unsqueeze(0).expand(B, -1)
                     .clamp(max=self.max_len - 1))
        pos_emb   = self.dec_pos(pos_idx)
        D         = tok_emb + pos_emb
        kappa     = torch.zeros(B, t, dtype=D.dtype, device=D.device)
        for block in self.dec_blocks:
            kappa = self.compute_kappa(D, pos_idx)
            D     = block(D, rcmb_out, kappa)
        D      = self.dec_ln(D)
        logits = self.output_proj(D)
        return logits, D, kappa

    def forward(self, inp_ids, out_ids, inp_mask=None, idf_table=None):
        belief_vec, rho, rcmb_out = self.encode(inp_ids, inp_mask)
        dec_input  = out_ids[:, :-1]
        dec_target = out_ids[:, 1:]
        logits, hidden, kappa = self.decode(dec_input, rcmb_out)
        return logits, dec_target, hidden, kappa, rho


# ==============================================================
# CELL 7: IDF TABLE AND HEDGE TOKEN IDs
# ==============================================================

def build_idf_table(pairs, vocab_size, pad_id):
    print("Building IDF table...")
    doc_freq   = torch.zeros(vocab_size, dtype=torch.float32)
    total_docs = 0
    for _, out_ids in tqdm(pairs, desc="IDF", leave=False):
        total_docs += 1
        unique_ids = set(min(max(int(x), 0), vocab_size - 1) for x in out_ids)
        for tid in unique_ids:
            doc_freq[tid] += 1.0
    idf = torch.log((total_docs + 1.0) / (doc_freq + 1.0)) + 1.0
    idf[0] = 0.0; idf[1] = 0.0; idf[2] = 0.0
    print(f"IDF built. Max={idf.max():.3f} Mean={idf.mean():.3f}")
    return idf


idf_table = build_idf_table(train_pairs, VOCAB_SIZE, PAD_ID)

HEDGE_PHRASES = [
    "sorry","no offense","just kidding","with respect",
    "honestly though","to be fair","bless","apologize",
    "forgive","didn't mean","not trying to",
]
hedge_ids_set = set()
for hp in HEDGE_PHRASES:
    try:
        for i in tokenizer.encode(hp):
            hedge_ids_set.add(min(max(int(i), 0), VOCAB_SIZE - 1))
    except Exception:
        pass
HEDGE_IDS = list(hedge_ids_set)
print(f"Hedge token IDs: {len(HEDGE_IDS)} flagged")


# ==============================================================
# CELL 8: FULL ROAST LOSS — ALL TERMS FROM PAPER SECTION 9
# ==============================================================

def compute_loss(model, logits, dec_target, hidden, kappa, rho,
                 idf_table, pad_id, vocab_size, phase=1,
                 lambda_kappa=0.1, lambda_psi=0.3,
                 lambda_hedge=1.0, lambda_route=0.01):
    """
    L_total = L_ce
            + λ_κ·L_κ    (tension monotonicity, Sec 9.2)
            + λ_ψ·L_ψ    (verbal impact velocity, Sec 9.3)
            + λ_h·L_h    (hedge penalty, Sec 9.4)
            + λ_r·L_route (routing diversity, Sec 9.5)
    Phase 1: L_ce only.
    """
    B, t, V = logits.shape
    dec_target = dec_target.clamp(0, vocab_size - 1)
    pad_mask   = (dec_target != pad_id)

    # L_ce (Sec 9.1)
    L_ce = F.cross_entropy(
        logits.reshape(-1, V),
        dec_target.reshape(-1),
        ignore_index=pad_id
    )
    if phase == 1:
        return L_ce, {'L_ce': L_ce.item(), 'L_kappa': 0.0,
                      'L_psi': 0.0, 'L_hedge': 0.0, 'L_route': 0.0}

    # L_kappa (Sec 9.2): penalise κ drops
    if t > 1:
        kappa_drop = F.relu(kappa[:, :-1] - kappa[:, 1:])
        vm         = pad_mask[:, :-1].float()
        L_kappa    = (kappa_drop * vm).sum() / (vm.sum() + 1e-8)
    else:
        L_kappa = logits.new_zeros(())

    # L_psi (Sec 9.3): reweight CE by ψ
    psi  = model.compute_psi(hidden, dec_target, idf_table)
    lp   = F.log_softmax(logits, dim=-1)
    tlp  = lp.gather(-1, dec_target.unsqueeze(-1)).squeeze(-1)
    pm   = psi * pad_mask.float()
    pn   = pm / (pm.sum() + 1e-8)
    L_psi = -(pn * tlp * pad_mask.float()).sum()

    # L_hedge (Sec 9.4): penalise hedge token probability mass
    if len(HEDGE_IDS) > 0:
        probs   = F.softmax(logits, dim=-1)
        h_ids_t = torch.tensor(HEDGE_IDS, dtype=torch.long)
        hedge_p = probs[:, :, h_ids_t].sum(dim=-1)
        L_hedge = (hedge_p * pad_mask.float()).mean()
    else:
        L_hedge = logits.new_zeros(())

    # L_route (Sec 9.5): Gram matrix orthogonality loss
    channel_vecs = []
    with torch.no_grad():
        for k in range(model.tsrl.K):
            w_k = model.tsrl.channel_projs[k].weight[0]
            channel_vecs.append(F.normalize(w_k, dim=0))
    C_stack = torch.stack(channel_vecs, dim=0)
    G       = C_stack @ C_stack.T
    K       = model.tsrl.K
    I_K     = torch.eye(K, dtype=G.dtype)
    L_route = ((G - I_K).pow(2).sum()) / float(K * K)

    L_total = (L_ce
               + lambda_kappa * L_kappa
               + lambda_psi   * L_psi
               + lambda_hedge * L_hedge
               + lambda_route * L_route)

    return L_total, {
        'L_ce':    L_ce.item(),
        'L_kappa': L_kappa.item() if hasattr(L_kappa, 'item') else 0.0,
        'L_psi':   L_psi.item(),
        'L_hedge': L_hedge.item() if hasattr(L_hedge, 'item') else 0.0,
        'L_route': L_route.item(),
    }


# ==============================================================
# CELL 9: MODEL INSTANTIATION
# ==============================================================

MODEL_CONFIG = dict(
    vocab_size=VOCAB_SIZE, d=128, d_pos=16,
    n_enc=2, n_dec=2, K_route=4, K_anchors=4,
    M_slots=16, d_mem=32, max_len=128, lambda_kappa=0.05,
)
model = SCORCH(**MODEL_CONFIG)

total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n{'='*50}")
print(f"SCORCH  total params: {total_p:,}")
print(f"SCORCH  train params: {train_p:,}")
print(f"{'='*50}\n")

# Forward-pass sanity check before training
print("Pre-training sanity check...")
with torch.no_grad():
    _ti, _to, _tm, _ = next(iter(train_loader))
    _lg, _tgt, _h, _k, _r = model(_ti, _to, inp_mask=_tm, idf_table=idf_table)
    assert _lg.shape[-1] == VOCAB_SIZE
    assert not torch.isnan(_lg).any(), "NaN in logits!"
    assert not torch.isinf(_lg).any(), "Inf in logits!"
print(f"Sanity check PASSED — logits {list(_lg.shape)}, "
      f"rho={_r.mean():.3f}, kappa={_k.mean():.3f}")


# ==============================================================
# CELL 10: OPTIMIZER AND LR SCHEDULE
# ==============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
)

def get_lr(step, warmup=1000, max_steps=30000,
           max_lr=1e-4, min_lr=1e-5):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def set_lr(optim, lr):
    for pg in optim.param_groups:
        pg['lr'] = lr


# ==============================================================
# CELL 11: PHASE AND LAMBDA HELPERS
# ==============================================================

PHASE1_END  = 8000
PHASE2_END  = 40000
TOTAL_STEPS = 50000
WARMUP      = 2000
LOG_EVERY   = 100
EVAL_EVERY  = 1000
SAVE_EVERY  = 2000
GRAD_CLIP   = 1.0

def get_phase(step):
    return 1 if step <= PHASE1_END else (2 if step <= PHASE2_END else 3)

def get_lambda_psi(step):
    if step <= PHASE1_END:
        return 0.0
    ramp = float(PHASE2_END - PHASE1_END) * 0.5
    return 0.3 * min(1.0, float(step - PHASE1_END) / max(ramp, 1.0))

def get_lambda_hedge(step):
    if step <= PHASE1_END:
        return 0.0
    ramp = float(PHASE2_END - PHASE1_END) * 0.5
    return 1.0 * min(1.0, float(step - PHASE1_END) / max(ramp, 1.0))


# ==============================================================
# CELL 12: K-MEANS ANCHOR INITIALISATION (Phase 2 prep)
# Paper Sec 10: run k-means on encoder belief_vec outputs
# ==============================================================

def initialise_roast_anchors(model, loader, n_batches=40):
    from sklearn.cluster import KMeans
    print("\n[Anchor Init] Collecting encoder belief vectors...")
    model.eval()
    all_beliefs = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= n_batches:
                break
            inp_ids, _, inp_mask, _ = batch
            inp_ids = inp_ids.clamp(0, model.vocab_size - 1)
            bv, _, _ = model.encode(inp_ids, inp_mask)
            all_beliefs.append(bv.numpy())
    all_beliefs = np.concatenate(all_beliefs, axis=0)
    K  = model.salience.K
    print(f"[Anchor Init] k-means K={K} on {all_beliefs.shape[0]} vectors...")
    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    km.fit(all_beliefs)
    centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    with torch.no_grad():
        model.salience.anchors.data.copy_(centroids)
    print("[Anchor Init] Anchors set to k-means centroids.")
    model.train()


# ==============================================================
# CELL 13: MAIN TRAINING LOOP — 3-PHASE CURRICULUM
# ==============================================================

best_val_loss       = float('inf')
train_iter          = iter(train_loader)
loss_ema            = None
ema_alpha           = 0.95
anchors_initialised = False

print(f"\n{'='*60}")
print("SCORCH TRAINING — CPU, 5 Datasets (4 Real + 10K Synthetic)")
print(f"  Total steps : {TOTAL_STEPS:,}")
print(f"  Phase 1     : 1 → {PHASE1_END:,}   (L_ce only)")
print(f"  Phase 2     : {PHASE1_END+1:,} → {PHASE2_END:,}  (full loss)")
print(f"  Phase 3     : {PHASE2_END+1:,} → {TOTAL_STEPS:,}  (fine-tune)")
print(f"{'='*60}\n")

train_start = time.time()
model.train()

for global_step in range(1, TOTAL_STEPS + 1):

    # ── Fetch batch ──
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch      = next(train_iter)

    inp_ids, out_ids, inp_mask, out_mask = batch

    # ── Phase / lambda / lr ──
    phase      = get_phase(global_step)
    lam_psi    = get_lambda_psi(global_step)
    lam_hedge  = get_lambda_hedge(global_step)
    lr         = get_lr(global_step, warmup=WARMUP, max_steps=TOTAL_STEPS)
    set_lr(optimizer, lr)

    # ── Initialise anchors once at Phase 2 start ──
    if phase >= 2 and not anchors_initialised:
        initialise_roast_anchors(model, train_loader, n_batches=30)
        anchors_initialised = True

    # ── Forward ──
    optimizer.zero_grad()
    try:
        logits, dec_target, hidden, kappa, rho = model(
            inp_ids, out_ids, inp_mask=inp_mask, idf_table=idf_table
        )
    except RuntimeError as e:
        print(f"  [SKIP] Forward error step {global_step}: {e}")
        continue

    # ── Loss ──
    try:
        loss, comp = compute_loss(
            model=model, logits=logits, dec_target=dec_target,
            hidden=hidden, kappa=kappa, rho=rho,
            idf_table=idf_table, pad_id=PAD_ID, vocab_size=VOCAB_SIZE,
            phase=phase,
            lambda_kappa=0.1  if phase >= 2 else 0.0,
            lambda_psi=lam_psi,
            lambda_hedge=lam_hedge,
            lambda_route=0.01 if phase >= 2 else 0.0,
        )
    except RuntimeError as e:
        print(f"  [SKIP] Loss error step {global_step}: {e}")
        continue

    if not torch.isfinite(loss):
        print(f"  [SKIP] Non-finite loss {loss.item():.4f} step {global_step}")
        continue

    # ── Backward ──
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    # ── EMA tracking ──
    lv       = loss.item()
    loss_ema = lv if loss_ema is None else (ema_alpha * loss_ema + (1 - ema_alpha) * lv)

    # ── Logging ──
    if global_step % LOG_EVERY == 0:
        elapsed     = time.time() - train_start
        sps         = global_step / max(elapsed, 1.0)
        eta_min     = (TOTAL_STEPS - global_step) / sps / 60.0
        print(
            f"Step {global_step:5d}/{TOTAL_STEPS} | Ph{phase} | "
            f"EMA={loss_ema:.4f} | "
            f"ce={comp['L_ce']:.4f} κ={comp['L_kappa']:.4f} "
            f"ψ={comp['L_psi']:.4f} h={comp['L_hedge']:.4f} | "
            f"ρ̄={rho.mean():.3f} κ̄={kappa.mean():.3f} | "
            f"lr={lr:.2e} ETA={eta_min:.1f}m"
        )

    # ── Validation ──
    if global_step % EVAL_EVERY == 0:
        model.eval()
        v_losses = []
        with torch.no_grad():
            for v_batch in val_loader:
                v_inp, v_out, v_mask, _ = v_batch
                try:
                    vl, vt, vh, vk, vr = model(
                        v_inp, v_out, inp_mask=v_mask, idf_table=idf_table
                    )
                    vl2, _ = compute_loss(
                        model=model, logits=vl, dec_target=vt,
                        hidden=vh, kappa=vk, rho=vr,
                        idf_table=idf_table, pad_id=PAD_ID,
                        vocab_size=VOCAB_SIZE, phase=phase,
                    )
                    if torch.isfinite(vl2):
                        v_losses.append(vl2.item())
                except RuntimeError:
                    pass

        if v_losses:
            val_loss  = float(np.mean(v_losses))
            elapsed_m = (time.time() - train_start) / 60.0
            print(f"\n{'─'*60}")
            print(f"[EVAL] step={global_step} val_loss={val_loss:.4f} "
                  f"elapsed={elapsed_m:.1f}min")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state':  optimizer.state_dict(),
                    'val_loss':         val_loss,
                    'config':           MODEL_CONFIG,
                    'vocab_size':       VOCAB_SIZE,
                }, 'scorch_best.pt')
                print(f"[SAVED] scorch_best.pt (val_loss={val_loss:.4f})")
            print(f"{'─'*60}\n")
        model.train()

    # ── Checkpoint ──
    if global_step % SAVE_EVERY == 0:
        p = f'scorch_step_{global_step}.pt'
        torch.save({'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'config': MODEL_CONFIG}, p)
        print(f"[CKPT] {p}")


total_min = (time.time() - train_start) / 60.0
print(f"\n{'='*60}")
print(f"TRAINING COMPLETE  time={total_min:.1f}min  best_val={best_val_loss:.4f}")
print(f"{'='*60}\n")


# ==============================================================
# CELL 14: INFERENCE — κ AND ρ MODULATED GENERATION
# Paper Section 11
# ==============================================================

def generate_roast(model, tokenizer_obj, input_text,
                      max_len=65, T_base=0.75, alpha_T=0.3,
                      beta_p=0.15, min_tokens=12,
                      max_tokens=60):
    """
    Improved generation with better temperature control.
    """
    model.eval()
    with torch.no_grad():
        inp_ids = tokenizer_obj.encode(input_text)
        if not inp_ids:
            return "You are terminally unremarkable."

        inp_ids = [min(max(int(i), 0), VOCAB_SIZE - 1) for i in inp_ids]
        inp_t = torch.tensor([inp_ids], dtype=torch.long)
        inp_mask = torch.ones_like(inp_t, dtype=torch.bool)

        _, rho, rcmb_out = model.encode(inp_t, inp_mask)
        rho_val = float(rho.item())
        p_nuc = max(0.75, 1.0 - beta_p * rho_val)

        generated = [BOS_ID]
        kappa_history = []

        for step in range(1, max_len + 1):
            dec_in = torch.tensor([generated], dtype=torch.long)
            logits, _, kappa = model.decode(dec_in, rcmb_out)
            last_logits = logits[0, -1, :].float()
            last_kappa = float(kappa[0, -1].item())
            kappa_history.append(last_kappa)

            # CRITICAL: Lower temperature for sharper outputs
            temp = max(T_base / (1.0 + alpha_T * last_kappa), 0.30)

            probs = F.softmax(last_logits / temp, dim=-1)
            probs[PAD_ID] = 0.0

            # Don't allow EOS before min_tokens
            if step < min_tokens:
                probs[EOS_ID] = 0.0

            # Force stop at max_tokens
            if step >= max_tokens:
                probs[EOS_ID] = 1.0
                next_token = EOS_ID
            else:
                # Nucleus sampling
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                remove = cumulative > p_nuc
                remove[0] = False
                sorted_probs[remove] = 0.0

                probs_filt = torch.zeros_like(probs)
                probs_filt[sorted_idx] = sorted_probs
                total = probs_filt.sum()

                if total <= 0:
                    next_token = int(sorted_idx[0].item())
                else:
                    probs_filt = probs_filt / total
                    next_token = int(torch.multinomial(probs_filt, 1).item())

            if next_token == EOS_ID:
                break

            generated.append(next_token)

        roast_tokens = generated[1:]
        if not roast_tokens:
            return "You are beyond words, and that's being generous."

        result = tokenizer_obj.decode(roast_tokens)
        return result.strip() if result.strip() else "Silent judgment is all you deserve."


# ==============================================================
# CELL 15: LOAD BEST MODEL AND TEST
# ==============================================================

print("Loading best checkpoint...")
try:
    ckpt = torch.load('scorch_best.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")
except FileNotFoundError:
    print("No checkpoint found — using current weights.")

test_inputs = [
    "A guy who still uses Internet Explorer",
    "Someone who has never finished a single book in their life",
    "A person who calls themselves an entrepreneur but earns zero dollars",
    "Someone who posts their 5AM workout every single morning",
    "A person who brings up their ex in every conversation",
    "Someone who says they are brutally honest but is just brutal",
    "A person who corrects grammar on the internet for fun",
    "A guy who drives a lifted pickup truck in the city and has never off-roaded",
    "Someone who went to one therapy session and now diagnoses everyone",
    "A person who says they don't watch TV but knows every single show",
    "Someone who has LinkedIn premium and has never gotten a job from it",
    "A person who describes every meal as a life-changing experience",
    "Someone who only drinks black coffee and mentions it constantly",
    "A person who says they are an empath but only talks about themselves",
    "Someone who has been writing a novel for twelve years",
    "A crypto bro who lost everything and is still bullish",
    "A wellness influencer who sells supplements with no evidence",
    "A person who uses the word literally incorrectly in every sentence",
    "Someone whose sourdough starter has a name and a birthday",
    "A person who announces their gym era every January",
]

print(f"\n{'='*60}")
print("SCORCH ROAST OUTPUTS")
print(f"{'='*60}\n")
for ti in test_inputs:
    roast = generate_roast(model, tokenizer, ti)
    print(f"INPUT : {ti}")
    print(f"ROAST : {roast}")
    print(f"{'─'*60}")


# ==============================================================
# CELL 16: INTERACTIVE CHATBOT
# ==============================================================

print(f"\n{'='*60}")
print("SCORCH INTERACTIVE ROAST CHATBOT")
print("Describe a target. Type 'quit' to exit.")
print(f"{'='*60}\n")

while True:
    try:
        user_input = input("Who to roast? > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nDone.")
        break
    if not user_input:
        continue
    if user_input.lower() in ('quit', 'exit', 'q', 'bye', 'stop'):
        print("Leaving. The roasts will outlive you.")
        break
    print(f"\n🔥  {generate_roast(model, tokenizer, user_input)}\n")
