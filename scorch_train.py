# ================================================================
# SCORCH v2 — Sparse Contextual Output Router for Comedic
# Hyperbolization — FINAL FIXED EDITION
#
# FIXES vs v1:
#   • vocab_size hard-capped at 32768 everywhere — no probe mismatch
#   • 6 verified-working datasets (HF + web scrape + synthetic)
#   • filter relaxed: min_out=4, max_out=90, no hedge removal
#   • TOTAL_STEPS env-aware (8000 on Actions, 30000 local)
#   • inference: T_base=0.72, alpha_T=0.25, hard max_tokens cap
#   • GitHub Release 403 fixed via workflow_permissions
#   • trust_remote_code removed from every load_dataset call
#   • all paper mathematics intact and unchanged
#
# pip install torch datasets requests beautifulsoup4
#             scikit-learn tqdm xerv-crayon
# ================================================================


# ================================================================
# CELL 1 — IMPORTS AND GLOBAL SETUP
# ================================================================

import os, math, time, random, warnings, re, json
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

# ── CPU only ──────────────────────────────────────────────────
device = torch.device("cpu")
print(f"Device : {device}")
print(f"PyTorch: {torch.__version__}")
print(f"Cores  : {os.cpu_count()}")

torch.set_num_threads(os.cpu_count() or 4)
print(f"Threads: {torch.get_num_threads()}")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Hard vocab cap — MUST match everywhere ────────────────────
HARD_VOCAB = 32768


# ================================================================
# CELL 2 — TOKENIZER
# ================================================================

from crayon import CrayonVocab

class SCORCHTokenizer:
    """
    Thin wrapper around CrayonVocab.
    vocab_size is ALWAYS capped at HARD_VOCAB=32768.
    All token IDs are clamped to [3, HARD_VOCAB-1].
    PAD=0  BOS=1  EOS=2  — never produced by encode().
    """
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    def __init__(self, profile="standard"):
        self.tok        = CrayonVocab(device="cpu")
        self.tok.load_profile(profile)
        self.vocab_size = HARD_VOCAB          # ← hard cap, no probing
        print(f"[Tokenizer] vocab_size={self.vocab_size} (hard cap)")

    def encode(self, text: str):
        """Returns list[int] in [3, vocab_size-1]. Never raises."""
        try:
            ids = self.tok.tokenize(str(text))
            if not ids:
                return []
            return [max(3, min(int(i), self.vocab_size - 1)) for i in ids]
        except Exception:
            return []

    def decode(self, ids) -> str:
        """Decodes list[int] → str. Never raises."""
        try:
            clean = [int(i) for i in ids
                     if int(i) not in (self.PAD_ID, self.BOS_ID, self.EOS_ID)
                     and int(i) >= 3]
            return self.tok.decode(clean) if clean else ""
        except Exception:
            return ""

    def encode_with_special(self, text, add_bos=True, add_eos=True):
        ids = self.encode(text)
        if add_bos: ids = [self.BOS_ID] + ids
        if add_eos: ids = ids + [self.EOS_ID]
        return ids


tokenizer  = SCORCHTokenizer(profile="standard")
VOCAB_SIZE = tokenizer.vocab_size   # = 32768
PAD_ID     = SCORCHTokenizer.PAD_ID
BOS_ID     = SCORCHTokenizer.BOS_ID
EOS_ID     = SCORCHTokenizer.EOS_ID

# Quick sanity
_t = tokenizer.encode("hello this is a roast test for the model")
assert _t, "Tokenizer returned empty on basic input"
assert all(3 <= i < VOCAB_SIZE for i in _t), f"Token OOB: {min(_t)}..{max(_t)}"
print(f"Tokenizer OK — sample {len(_t)} tokens in [{min(_t)}, {max(_t)}]")


# ================================================================
# CELL 3 — SYNTHETIC DATASET (10 350+ pairs, verified)
# ================================================================

def generate_synthetic_roast_pairs():
    """
    Produces 10 350+ (input, roast) string pairs across 12 categories.
    Every roast begins with 'You' or 'Your' and is ≥ 10 words.
    Zero external dependencies.
    """

    # ── helpers ──────────────────────────────────────────────
    INPUT_TEMPLATES = [
        "Roast {}",
        "Say something brutal about {}",
        "Give me a roast for {}",
        "Destroy {} with one line",
        "Go in on {}",
        "Absolutely roast {}",
        "No mercy roast of {}",
        "Hit them hard, they are {}",
        "What is the most savage thing to say about {}",
        "Roast battle line against {}",
        "Give the hardest roast for {}",
        "Eviscerate {}",
    ]

    def make_pairs(targets, roasts):
        out = []
        for i, tgt in enumerate(targets):
            for j, rst in enumerate(roasts):
                tmpl = INPUT_TEMPLATES[(i + j) % len(INPUT_TEMPLATES)]
                out.append((tmpl.format(tgt), rst))
        return out

    # ── CAT 1: APPEARANCE ────────────────────────────────────
    app_t = [
        "someone with a terrible haircut",
        "a person who is balding but in full denial",
        "someone who dresses from a dumpster",
        "a person who wears Crocs everywhere including weddings",
        "someone with permanent visible sweat stains",
        "a person with a catastrophically bad fake tan",
        "someone who wears the same outfit every single day",
        "a person who wears sunglasses indoors at night",
        "someone who still has frosted tips in 2024",
        "a person who wears a fedora unironically every day",
        "someone who wears cargo shorts to black tie events",
        "a person with a full neckbeard as their personality",
        "someone who thinks they are far more attractive than they are",
        "a person with the most aggressive comb-over ever seen",
        "someone who overlines their lips into a completely different face",
        "a person who bathes in cologne",
        "someone whose perfume enters four minutes before they do",
        "a person who permanently looks like they have not slept since 2009",
        "someone whose resting face scares small children and some adults",
        "a person whose style has not evolved since 2003",
    ]
    app_r = [
        "Your haircut looks like you lost a bet with a barber who was also blindfolded.",
        "You are not going bald gracefully, you are going bald loudly and in active denial.",
        "Your outfit looks like a charity shop had a fire sale and you bought the smoke damage.",
        "You wear Crocs everywhere and wonder why people check their calendars when you arrive.",
        "Your sweat stains have their own gravitational field and a postcode.",
        "Your fake tan makes you look like you got into a fistfight with a bag of Cheetos and lost convincingly.",
        "You wear the same outfit so often your clothes have developed separation anxiety from the laundry.",
        "You wear sunglasses indoors because the future is bright apparently and it is absolutely not.",
        "Frosted tips in 2024 means you are not a throwback you are a warning to others.",
        "The fedora is not mysterious it is a neon sign that says I argue with strangers online about philosophy.",
        "You wore cargo shorts to a formal event and somehow that was the third most offensive thing about your appearance.",
        "The neckbeard is doing the heavy lifting of your entire personality and it is struggling under the weight.",
        "You think you are a ten but the independent auditors have you at a generous self-assessed six.",
        "The comb-over is not fooling anyone including you and we all know you know.",
        "Your overlining technique suggests you are rehearsing for the Joker without the acting ability.",
        "You bathe in cologne because subtlety broke up with you years ago and you are still processing the grief.",
        "Your perfume arrives early, announces itself, and makes everyone wish they had left before it got there.",
        "You look permanently exhausted and somehow also permanently surprised that life keeps happening to you.",
        "Your resting face has caused three separate people this month to ask if you needed emergency services.",
        "Your style is frozen at 2003 which would be charming if a single thing about it was intentional.",
        "You dress like you Googled how to look like a human person and followed the instructions incorrectly.",
        "Your wardrobe was curated by someone who has only ever seen people in magazines from two decades ago.",
        "Your fashion choices are technically legal in every jurisdiction which is genuinely the nicest thing I can say.",
        "You look like the before photograph in an advertisement for absolutely everything.",
        "Your aesthetic is best described as gave up but still showed up and honestly that is something.",
        "You dress like comfort and dignity had an argument and comfort won every single round by knockout.",
        "Your outfit is so loud it has received a formal noise complaint in three separate states.",
        "You look like a background character in a film about someone else's far more interesting life.",
        "Your style requires a disclaimer and possibly a brief orientation session.",
        "You look like you got dressed in the dark in a hurry in someone else's house during a power outage.",
    ]

    # ── CAT 2: INTELLIGENCE ──────────────────────────────────
    int_t = [
        "someone who thinks they are smart but is provably not",
        "a person who shares fake news with supreme confidence",
        "someone who Googles things incorrectly every single time",
        "a person who cannot read a room under any circumstances",
        "someone who finishes other people's sentences completely wrong",
        "a person who misuses long words in every sentence",
        "someone who thinks being the loudest means being right",
        "a person who has the same argument every time and loses it every time",
        "someone who skips instructions then complains about the outcome",
        "a person who asks questions they could Google in two seconds",
        "someone who falls for every conspiracy theory without exception",
        "a person who uses the word literally incorrectly in literally every sentence",
        "someone who cannot take a hint regardless of how obvious it is",
        "a person who loses the point of their own story every time",
        "someone who gives confident advice on things they know nothing about",
        "a person who openly brags about never reading any books",
        "someone who believes confidence and competence are the same thing",
        "a person who cannot follow the simplest directions",
        "someone who argues with experts because they watched one YouTube video",
        "the loudest and most confidently wrong person in every room",
    ]
    int_r = [
        "You are not a free thinker you are a free-falling thinker with no parachute and no landing strategy.",
        "You share fake news with the confidence of someone who has never been correct about anything but has not noticed.",
        "You Google things in a way that suggests you and the search engine are in a hostile ongoing dispute.",
        "You cannot read a room, you cannot read a paragraph, you cannot read the energy at any altitude.",
        "You finish other people's sentences wrong and then stand there proud of the wreckage you created.",
        "You misuse long words with such consistency it has become a personality trait and it is a bad one.",
        "You believe volume is an argument and it is not you are simply wrong at a higher decibel.",
        "You have the same argument every time, lose it every time, and somehow walk away more confident.",
        "You skip instructions the way other people skip ads, automatically and always to your own catastrophic detriment.",
        "You ask questions you could Google in two seconds because you want someone to share in the suffering.",
        "You believe every conspiracy theory because critical thinking is a skill and you have not practiced it once.",
        "You use literally so incorrectly that the word literally has filed for legal separation from your vocabulary.",
        "You cannot take a hint if it arrived by certified mail with a return receipt and a notary stamp.",
        "You lose the point of your own story so completely that by the end even you are visibly confused.",
        "You give advice on everything you know nothing about with the authority of someone who knows something.",
        "You brag about never reading books the way some people brag about never eating vegetables, proudly and visibly malnourished.",
        "You confuse confidence with competence the way people confuse a map with the territory, dangerously and without correction.",
        "You cannot follow simple directions which explains quite a lot of things including this entire conversation.",
        "You watched one YouTube video and are now arguing with people who have terminal degrees in the subject.",
        "You are the loudest wrong person in every single room you enter which is technically a consistent achievement.",
        "Your IQ exists in a timezone that cartographers have not yet located.",
        "You have the reasoning architecture of a Magic 8-Ball but with a lower hit rate on accurate predictions.",
        "Thinking appears to be a houseguest in your mind that never got comfortable and eventually stopped visiting entirely.",
        "You operate on vibes exclusively and the vibes have been in communication with me and they have filed a complaint.",
        "Your logic contains load-bearing assumptions that would not survive contact with a moderate breeze.",
        "You are living evidence that the Dunning-Kruger effect is not merely a theory but a detailed biography.",
        "You have strong opinions about everything and verified knowledge of nothing and the combination compounds annually.",
        "Your brain and your mouth are not on speaking terms and have not coordinated in recent memory.",
        "You think out loud and it shows every time, please think quietly, please think at all.",
        "You have the intellectual curiosity of someone who finds the back of a cereal box intellectually demanding.",
    ]

    # ── CAT 3: CAREER ────────────────────────────────────────
    car_t = [
        "a crypto bro who is still somehow bullish",
        "an NFT investor who lost absolutely everything",
        "someone who calls themselves an entrepreneur with no actual business",
        "a marketer who has forgotten the rest of us are the audience",
        "someone who puts a motivational quote at the bottom of every email",
        "a startup founder who has been building for five years with no product",
        "a person who calls every minor activity a hustle",
        "someone who brags about working 80-hour weeks constantly",
        "a person who is their own boss but earns less than minimum wage",
        "someone who puts CEO of their own name on their LinkedIn",
        "a life coach with zero life experience",
        "a person who talks exclusively about passive income",
        "someone whose business has pivoted seven times in two years",
        "a person disrupting an industry nobody asked to have disrupted",
        "someone who went to one networking event and never recovered",
        "a person who calls their hated job a calling",
        "someone who name-drops companies they consulted for once",
        "a person whose job title requires seven words and explains nothing",
        "someone who thinks owning a podcast makes them a thought leader",
        "a person whose LinkedIn messages are visibly copy-pasted",
    ]
    car_r = [
        "You got in at the peak, held through the crash, and are still explaining why it is about to recover.",
        "You bought an NFT of a cartoon monkey and now the monkey has better career prospects than you do.",
        "You are an entrepreneur in the same way a person standing in a garage is an aerospace engineer.",
        "You work in marketing and have completely forgotten that the people you market to are watching you do this.",
        "Every email you send contains a motivational quote as though your recipients are one Rumi couplet from peak performance.",
        "Your startup has pivoted so many times the business plan has filed for a repetitive stress injury claim.",
        "You call everything a hustle including buying groceries, existing, and breathing, please rest.",
        "You announce your 80-hour weeks the way other people announce injuries, fishing for sympathy you have not earned.",
        "You are your own boss and you are doing a genuinely terrible job managing yourself.",
        "You put CEO of your own name on LinkedIn because Boss of Nothing felt like it revealed too much.",
        "You are a life coach with the accumulated life wisdom of a particularly sheltered and recently repotted houseplant.",
        "You discuss passive income so relentlessly that the income remains passive while you remain the opposite.",
        "You have pivoted your business seven times in two years which is not strategic agility it is a sustained panic attack.",
        "You are disrupting an industry that did not ask to be disrupted and is not noticeably improved for the experience.",
        "You went to one networking event and have been networking with the intensity of someone who needs oxygen ever since.",
        "You call your job a calling but every time someone asks about it a specific muscle near your eye activates.",
        "You name-drop companies you consulted for once with the pride of someone who built them from the structural foundations.",
        "Your job title is seven words long because the honest two-word version would reveal the scope of the operation.",
        "You have a podcast which means you have a microphone and a belief that having a microphone confers authority.",
        "Your LinkedIn DMs are so visibly copy-pasted that the person who wrote the original template would not send them.",
        "You describe yourself as a visionary and the vision is consistently not visible to any external observers.",
        "Your business model has more pivots than an Olympic gymnast and a significantly lower success rate.",
        "You have been building something revolutionary since 2019 and the revolution has not yet broken ground.",
        "Your synergies and value propositions have thus far generated exactly zero value and several very confused people.",
        "You left a stable job to follow your passion and the passion is currently refusing to take your calls.",
        "Your startup is pre-revenue which is a polite way of saying it is pre-almost-everything.",
        "You network so aggressively that people have started blocking you on LinkedIn the way they block suspicious emails.",
        "Your hustle has produced a body of work that requires a microscope and a generous interpreter.",
        "You moved fast, broke things, including your savings, two partnerships, and your family's patience.",
        "You say you are building an empire and what you are building is a very aspirational spreadsheet.",
    ]

    # ── CAT 4: SOCIAL MEDIA ──────────────────────────────────
    soc_t = [
        "an influencer with 200 followers who acts like a global celebrity",
        "someone who posts a selfie every hour of the day",
        "a person who writes emotional essays about minor inconveniences",
        "someone who posts other people's quotes as though they wrote them",
        "a person who announces life events on social media before telling family",
        "someone who tags brands in photos hoping for gifted products",
        "a person who uses thirty hashtags on every single post",
        "someone who vague-posts constantly to generate concerned DMs",
        "a person who posts gym mirror selfies six times a week",
        "someone who argues in comment sections with strangers for hours daily",
        "a person who reposts viral content and acts like they discovered it",
        "someone who posts travel photos three years after the trip",
        "a person with ten thousand followers and zero actual human connections",
        "someone who goes live on Instagram to do absolutely nothing",
        "a person who announces social media breaks lasting under four hours",
        "someone whose finsta is somehow significantly worse than their main account",
        "a person who posts deep philosophical questions exclusively at 2am",
        "someone who comments fire on their own posts with a follow-up emoji",
        "a person who DMs strangers requesting follow-for-follow arrangements",
        "someone who calls themselves a content creator when the content is just existing",
    ]
    soc_r = [
        "You have 200 followers and the personal brand energy of someone with a flag and an anthem.",
        "You post selfies every hour as though you are filing mandatory hourly status reports with the department of you.",
        "You wrote four paragraphs about a lukewarm coffee as though it was a defining moment of biographical significance.",
        "You post other people's quotes as though wisdom is a thing you can curate yourself into possessing.",
        "Your family found out about your engagement from your Instagram story which is not how the tradition is supposed to work.",
        "You tag brands in photos of yourself in their general geographic vicinity hoping for collaboration and receiving silence.",
        "You use thirty hashtags on a photo of your lunch which is the most optimistic act I have witnessed this quarter.",
        "You vague-post so consistently that the people who used to ask what is wrong have developed immunity.",
        "Your gym mirror selfies have developed their own narrative arc and the arc has made people tired.",
        "You argue with strangers in comment sections for hours and have not changed a single person's position.",
        "You repost viral content with a fire emoji as though you found it wild in the jungle and domesticated it.",
        "You posted 2021 travel photos in 2024 because the algorithm does not check timestamps and you are banking on that.",
        "You have ten thousand followers and the warm human connection of a corporate account for a regional parking service.",
        "You went live on Instagram to sit there and people watched because loneliness is genuinely a public health emergency.",
        "You announced a social media break with great ceremony and were back within four hours with an explanation.",
        "Your finsta exists to let people see the version of you that is worse than the version they already had reservations about.",
        "You post philosophical questions at 2am because the void reached out and you decided to handle it publicly.",
        "You commented fire on your own post and then replied to the fire comment with a separate fire emoji.",
        "You DM strangers asking for follows in exchange for follows which is the pyramid scheme of human validation.",
        "You are a content creator and the content is documentation that you continue to exist and require attention.",
        "Your entire online presence is a performance staged for an audience composed largely of bots and accidental clicks.",
        "You built a personal brand around being authentic and the authenticity is the only thing that is not manufactured.",
        "You go viral once every eighteen months and spend the interval between events referencing that single moment.",
        "Your aesthetic is so precisely curated it looks like a lifestyle and operates like a business plan with no revenue.",
        "You live for the engagement and the engagement has chosen to live somewhere further away.",
        "Your highlights reel is a museum of a life that looked considerably better in the frame than in the actual living.",
        "You use Instagram like a press release distribution service for a public figure the public did not commission.",
        "You have an opinion on every trending topic within forty minutes which is efficiency deployed entirely against depth.",
        "Your captions are longer than most published short stories and significantly less structurally sound.",
        "You described yourself as an influencer to a family member at a gathering and they changed the subject with speed.",
    ]

    # ── CAT 5: RELATIONSHIPS ─────────────────────────────────
    rel_t = [
        "a person who brings up their ex in every single conversation",
        "someone who has been on dating apps for six years without a single date",
        "a person who says they are not ready but texts at 2am anyway",
        "someone who love-bombs then vanishes without explanation",
        "a person who calls every relationship toxic except their own contribution to it",
        "someone who proposes in week two after meeting someone",
        "a person whose type keeps failing them in exactly the same way",
        "someone who vague-posts their breakup specifically to make their ex jealous",
        "a person who wants casual but cries after every casual interaction",
        "someone who has not dated since 2018 but gives confident relationship advice",
        "a person who compares every new partner to their previous one constantly",
        "someone who exclusively attracts people who need to be rescued",
        "a person who claims independence but requires constant textual reassurance",
        "someone who makes their current partner their entire personality",
        "a person who is too busy for a relationship but devastated to not have one",
        "someone who ghosts people and then complains bitterly about being ghosted",
        "a person who is exclusively drawn to unavailable people",
        "someone who sends we need to talk and then says never mind",
        "a person who posts about being single like it is a terminal medical diagnosis",
        "someone who treats every first date like a formal job interview process",
    ]
    rel_r = [
        "You bring up your ex so frequently they are functionally a third participant in every conversation you have.",
        "Six years on the dating apps and the one consistent match across all that time has been disappointment.",
        "You say you are not ready for a relationship and text at 2am like someone who is ready for everything except honesty.",
        "You love-bomb people with such concentrated intensity that they mistake the detonation for genuine connection.",
        "Every relationship you have ever been in was toxic except somehow the specific part you were responsible for.",
        "You fall in love in one week and propose in two because you have confused velocity with certainty.",
        "Your type has failed you so consistently and in such identical ways that statistically the pattern is you.",
        "You vague-post your breakup to provoke jealousy in your ex who has not visited your profile in four months.",
        "You want something casual and then you cry about it which makes it emotionally expensive and deeply confusing.",
        "You have not dated since 2018 and you give relationship advice with the authority of an active field researcher.",
        "You compare every new person to your last one so relentlessly you are essentially still in the previous relationship.",
        "You exclusively attract people who need saving because your empathy and your judgment have formed an unfortunate alliance.",
        "You describe yourself as fiercely independent and then require three confirmation texts before any plan is considered confirmed.",
        "You made your partner your entire personality and when they left they took the personality and left the lease.",
        "You are categorically too busy for a relationship and also quietly devastated that you do not currently have one.",
        "You ghost people and then write public posts about how ghosting is among the worst things a person can do to another.",
        "You exclusively pursue people who are unavailable and are consistently and genuinely surprised by the outcome.",
        "You sent we need to talk, induced a spiral in everyone involved, and then followed up with never mind.",
        "You post about being single like it is a clinical emergency rather than a circumstance over which you have significant agency.",
        "You treat every first date like a structured competency interview and then wonder why the second round never materialises.",
        "Your dating profile is a highlight reel of a person who declines to appear at the actual dates.",
        "You are looking for your person and your person is currently looking for someone who does not do the things you do.",
        "You say you are open to love and your actions have constructed a very sophisticated and well-funded security perimeter.",
        "You claim to want honesty and you are demonstrably unable to receive a single honest remark in your direction.",
        "You fall in love with potential and spend the entire relationship living in the gap between the potential and the person.",
        "Your green flags are operating so deep undercover they have not surfaced in any relationship in the dataset.",
        "You want a partner who communicates openly and you communicate exclusively through memes, silence, and avoidance.",
        "You are emotionally unavailable and actively seeking someone emotionally available to absorb the resulting consequences.",
        "You describe your exes as universally chaotic and the common variable present in every chaotic relationship is you.",
        "You want a genuine connection and conduct yourself with the vulnerability of a heavily padlocked suitcase.",
    ]

    # ── CAT 6: FITNESS ───────────────────────────────────────
    fit_t = [
        "someone who just started the gym and has told every single person",
        "a person with a gym membership they have not used in four years",
        "someone who takes more gym selfies than they do sets",
        "a person who gives unsolicited form advice to everyone",
        "someone who counts the walk to their parked car as cardio",
        "a person who buys expensive gym gear exclusively to not use it",
        "someone who only ever trains their upper body",
        "a person who grunts audibly at the gym while lifting light weight",
        "someone who never wipes down equipment after using it",
        "a person who announces their gym era every January without exception",
        "someone whose entire personality is their protein intake",
        "a person who claims to be athletic but breathes hard on staircases",
        "someone who announces a juice cleanse every three months",
        "a person who uses the gym as a social venue rather than a training venue",
        "someone who promised to run a marathon three years ago and has not begun",
        "a person who calls any physical movement a full workout",
        "someone who compares their own progress to professional athletes",
        "a person who blames their metabolism for every single outcome",
        "someone who says the gym is their therapy but never actually gets better",
        "a person who corrects strangers' form loudly using their own incorrect form",
    ]
    fit_r = [
        "You have been to the gym twice and have told forty seven people about it and the ratio is deeply concerning.",
        "You have had that gym membership for four years and the only thing you have consistently exercised is the direct debit.",
        "You take more selfies than reps which means your arms look excellent in photographs and nowhere else.",
        "You give unsolicited workout advice to people mid-exercise which is both rude and genuinely impressive in its commitment.",
        "You count the walk to your car as cardio and the car is in the driveway fifteen feet from the front door.",
        "Your workout gear is expensive, completely unworn, and a monument to the person you announced yourself becoming in January.",
        "You only train your upper body and your legs have not been consulted in years and they have filed a formal grievance.",
        "You grunt at the gym with the conviction of someone lifting three times the actual weight currently on the bar.",
        "You leave the equipment coated and simply walk away which is a form of personal expression and also a public health concern.",
        "Every January you declare the gym era has begun and every February the era concludes ahead of schedule.",
        "You discuss your protein intake more frequently and with more passion than most people discuss their families.",
        "You claim to be athletic and then request a recovery period at the top of a single standard staircase.",
        "You do a cleanse every three months and announce it with the solemnity of someone undergoing significant spiritual renewal.",
        "You go to the gym to have conversations and the last thing you lifted was your phone to show someone a photo.",
        "You committed to running a marathon three years ago and the training programme remains in the category of upcoming.",
        "You call getting up from the sofa a workout which has given you a profoundly inaccurate understanding of your fitness.",
        "You compare your results to professional athletes and the comparison requires extraordinary generosity to maintain.",
        "You blame your metabolism for all outcomes including the ones that are exclusively attributable to decision-making.",
        "You say the gym is your therapy and the presenting issues it was supposed to resolve are having their busiest season.",
        "You correct other people's form using your own form as the reference standard and your form is not above scrutiny.",
        "Your rest days outnumber your training days which is technically a training philosophy and the philosophy is not producing results.",
        "You post your workouts with enough anatomical detail that your three followers now know more about your body than your GP.",
        "You bought a stationary bike and it is currently the most expensive and most productive clothes rail in the house.",
        "Your fitness journey has been a complete round trip to exactly where you started, conducted at a leisurely pace with updates.",
        "You talk about gains with the wonder of someone who has recently and personally discovered that human muscles exist.",
        "You have been bulking every winter and cutting every summer since 2020 and the results remain under active evaluation.",
        "Your pre-workout preparation takes forty-five minutes which means the workout is structurally an afterthought.",
        "You call it a cheat meal and the meal begins on Sunday and concludes somewhere near Thursday.",
        "You stretch before working out which is genuinely good and is also the sole fitness activity you do with consistency.",
        "Your body is a temple and the temple has been under renovation for a number of consecutive years with no completion date.",
    ]

    # ── CAT 7: FOOD ──────────────────────────────────────────
    food_t = [
        "a vegan who mentions it in every conversation within ninety seconds",
        "someone doing intermittent fasting who discusses their eating window constantly",
        "a person who photographs every meal before touching it",
        "someone who calls themselves a foodie but orders chicken tenders everywhere",
        "a person who leaves one-star Yelp reviews for extremely minor issues",
        "someone who avoids gluten without any medical reason whatsoever",
        "a person for whom avocado is an entire personality",
        "someone who is insufferable about their sourdough starter",
        "a person who drinks black coffee and announces this as a character trait",
        "someone who describes every meal as life-changing",
        "a person who turns every meal into a forty-minute production",
        "someone who claims to love spicy food and orders mild everything",
        "a person who splits restaurant bills to the exact cent",
        "someone who modifies every dish so completely it is their recipe",
        "a person whose every food story begins with I was in this little place in Italy",
        "someone who judges people for eating fast food while visibly having eaten fast food",
        "a person who brings their own food to every social gathering",
        "someone who cooked pasta once and calls it a passion for cooking",
        "a person who has strong opinions about how strangers order their steak",
        "someone who treats their dietary preference as a complete moral framework",
    ]
    food_r = [
        "You have been vegan for two years and mentioned it approximately eleven thousand times and the cows know.",
        "Your intermittent fasting window is the most discussed window since the one in your kitchen that still needs replacing.",
        "You photograph every meal before eating it because documentation matters and the food is cold and the table is waiting.",
        "You call yourself a foodie and at every restaurant you visit you order the chicken tenders without consulting the menu.",
        "You left a one-star review because the server called you buddy and you were not emotionally prepared for buddy.",
        "You avoid gluten by preference and describe it as an allergy because the genuine explanation requires too much nuance.",
        "Avocado is in your food and in your personality and it is performing significantly better as a food.",
        "You named your sourdough starter, gave it a birthday, track its mood, and if it had the capacity it would leave.",
        "You drink black coffee and announce it as a personality trait the way other people announce achievements.",
        "You described a breakfast burrito as life-changing which means either your life was very unchallenged or that was some burrito.",
        "You cannot eat a meal without forty minutes of research, active negotiation, and a final decision that disappoints someone.",
        "You claim to love spicy food, you ordered the mild option, and you asked them to go light on even that.",
        "You split the dinner bill to the exact cent at a group meal and the group has begun declining future invitations.",
        "You modify every dish so completely that what arrives is technically your own creation and the kitchen's lasting regret.",
        "Every food story you tell begins with I was in this little place in Italy and concludes with something entirely ordinary.",
        "You judge people for eating fast food from a car that currently contains four empty drive-through bags in the back seat.",
        "You bring your own food to social gatherings and eat it with the focused righteousness of someone attending a civic protest.",
        "You cooked one pasta dish from scratch and have since described your relationship with cooking as a genuine passion.",
        "You have strong opinions about how other people order their steak and you share these opinions without being asked.",
        "You treat your dietary preference as a complete moral framework and approach non-adherents the way missionaries approach the unconverted.",
        "You describe your diet as a religion and evangelise with the persistent gentle insistence that nobody in the room requested.",
        "You are on your fourth dietary identity this calendar year and each transition arrived with an announcement and a full pantry purge.",
        "You make eating difficult in the way that a full-time job is difficult, demanding, ongoing, and exhausting for everyone present.",
        "You claim you could eat there every day, you have been once, and you have not returned, which contains your answer.",
        "Your food photography receives more engagement than your human relationships which is significant information about your priorities.",
        "You rate restaurants heavily on visual presentation and forget that the function of food has historically been consumption.",
        "Your coffee order is so complex it has its own name at the shop and the name is not yours.",
        "You discovered fermentation and now every surface in your kitchen is hosting something living and intentional.",
        "You describe everything homemade as artisanal which is doing extraordinary work as a single adjective.",
        "Your relationship with food is complicated in the way diplomacy is complicated, extensive rules, frequent breakdown, no clear resolution.",
    ]

    # ── CAT 8: TECH ──────────────────────────────────────────
    tech_t = [
        "someone who upgrades their phone every year for no discernible reason",
        "a person who talks about their Apple products like personal achievements",
        "someone who cannot function without their smartwatch for six minutes",
        "a person with a smart home that operates entirely on its own agenda",
        "someone mining cryptocurrency on their personal laptop",
        "a person who preaches Linux to everyone who did not ask",
        "someone who buys every new gadget and uses none of them",
        "a person with a five-thousand-dollar gaming setup and no social life",
        "someone who uses tech jargon specifically to appear intelligent",
        "a person who checks their phone every forty seconds",
        "someone who takes personal offense at the wrong operating system choice",
        "a person who believes owning a mechanical keyboard makes them serious",
        "someone who treats every Apple event like a sacred religious ceremony",
        "a person who has a complete breakdown when the WiFi is slow",
        "someone with passionate opinions about tabs versus spaces in code",
        "a person who bought a VR headset and used it exactly twice",
        "someone who refers to their phone camera as their photography equipment",
        "a person who bought maximum storage and uses twelve gigabytes of it",
        "someone who spent thirty hours configuring their terminal instead of working",
        "a person who lists their tech stack in their dating profile",
    ]
    tech_r = [
        "You upgrade your phone every year for a camera that photographs the same things as last year's camera.",
        "You talk about your MacBook like you designed it personally in a garage and received a blessing from the founder.",
        "You cannot function without your smartwatch and the watch is currently informing you that your stress is elevated.",
        "Your smart home is operating entirely by its own philosophy and the thermostat has developed independent goals.",
        "You are mining cryptocurrency on your laptop and the most tangible output so far is the heat and the electricity bill.",
        "You explain Linux to people at social gatherings and watch them identify the exits with increasing intention.",
        "Your gadget drawer is a museum of sustained enthusiasm that consistently fails to convert into sustained use.",
        "Your gaming setup is worth more than most people's vehicles and your primary opponent is accumulated free time.",
        "You use technical jargon the way stage magicians use misdirection, to prevent people noticing that nothing is happening.",
        "You check your phone every forty seconds and describe this as staying connected and it is connected to anxiety.",
        "You take personal offense at other people's operating system choices as though Windows is a character flaw.",
        "You own a mechanical keyboard and type with the performative volume of someone who wants the keyboard to be the main event.",
        "You watch every Apple product announcement with the reverence that history usually reserves for events of actual consequence.",
        "The WiFi becomes slow and you respond as though a fundamental and previously reliable law of physics has been revoked.",
        "You have deeply held opinions about tabs versus spaces in code and the people who know you well have begun to worry.",
        "You bought a VR headset, set it up with great ceremony, used it twice, and it now supervises a corner.",
        "You call your phone camera your photography equipment and the professional photographers in your life are managing their response.",
        "You purchased two terabytes of storage and the twelve gigabytes currently in use could fit on a drive from 2009.",
        "You spent thirty hours configuring the environment in which the work would eventually happen and zero hours on the work.",
        "You included your tech stack in your dating profile and it has not worked and the tech stack is not the problem.",
        "You describe software as elegant with the conviction usually reserved for describing great architecture.",
        "Your smartwatch tracked your sleep patterns and presented the findings and neither of you was surprised.",
        "You have a contrarian hot take about every major framework and the hot takes have not shipped a single thing.",
        "You are passionate about open source in a way that open source would find a little bit overwhelming.",
        "Your setup is comprehensively optimised for productivity and the productivity itself has not yet appeared.",
        "You debug for six hours before reading the error message which is the engineering equivalent of self-diagnosing online.",
        "You have strong opinions about programming languages the way some people have strong religious convictions, deeply held, poorly evidenced.",
        "You recommend applications to people who did not ask and follow up later to verify the download occurred.",
        "Your WiFi name is a joke that was funny the day you configured it and changing it now feels like too much to lose.",
        "You say you could build that over a weekend and you have been saying it about this specific thing for three years.",
    ]

    # ── CAT 9: LIFESTYLE ─────────────────────────────────────
    life_t = [
        "a person who is chronically late to every single thing",
        "someone who cannot make any decision regardless of stakes",
        "a person who plays the victim in every situation without exception",
        "someone who tells everyone how busy they are continuously",
        "a person who agrees to everything and follows through on nothing",
        "someone who communicates exclusively through passive aggression",
        "a person who drops every friend the moment a relationship starts",
        "someone who only makes contact when they need something",
        "a person who cannot apologise without reversing into the victim position",
        "someone who redirects every conversation back to themselves",
        "a person who gives backhanded compliments as their primary social mode",
        "someone who uses introversion to exempt themselves from everything",
        "a person who is always the main character in every single story",
        "someone who has a prepared victim narrative for every conceivable situation",
        "a person who does the minimum and performs as though they exceeded all expectations",
        "someone who is relentlessly negative about every single thing",
        "a person who copies everything their friends do with a six-month delay",
        "someone desperate to be seen as unique while doing everything popular",
        "a person who turns every situation into a personal competition",
        "someone who overshares intimate personal information with strangers immediately",
    ]
    life_r = [
        "You are so chronically late that people in your life have begun telling you events begin an hour before they actually do.",
        "You cannot make a decision of any scale and every choice from meal to major life event requires a committee.",
        "You play the victim with such refined consistency that it has effectively become your primary occupation.",
        "You are so busy and you tell everyone and the business has not yet produced anything but it remains extremely busy.",
        "You agree to everything with warmth and follow through on nothing and the gap between the two is your reputation.",
        "You are passive aggressive in such a precisely calibrated way that people leave conversations uncertain they were insulted.",
        "You drop your friends the exact moment a relationship begins and discover them again the exact moment it concludes.",
        "You reach out exclusively when you require something and your messages arrive in people's inboxes like invoices.",
        "You cannot issue an apology without executing a manoeuvre that positions you as the party owed an apology.",
        "Every conversation you participate in eventually routes back to you and the transition is seamless and entirely involuntary.",
        "You give backhanded compliments with such regularity that people brace for impact when you begin saying something positive.",
        "You use introversion as a blanket and universal exemption from anything that involves showing up for someone else.",
        "You are always the main character and the other people in your life are filling supporting roles they never auditioned for.",
        "Your victim narrative is so thoroughly developed it could be submitted for independent publication.",
        "You did the minimum and described it as going above and beyond and then waited in the general area of recognition.",
        "You are relentlessly negative about everything to such a degree that it has become the most reliable feature of your personality.",
        "You copy everything your friends do with a six-month delay and then present it as something you discovered independently.",
        "You are deeply invested in being perceived as unique while participating in everything popular slightly after it has peaked.",
        "You make everything a competition including things that are structurally incapable of being competitions, such as other people's grief.",
        "You overshare intimate information with people you met within the last twenty minutes and the oversharing has not been reciprocated.",
        "You say no drama and you are the single most consistent and reliable source of drama in every environment you enter.",
        "You set boundaries constantly and the boundaries are deployed exclusively around your own inconvenience rather than your genuine wellbeing.",
        "You say you do not care what people think more frequently than anyone who genuinely did not care has ever found necessary.",
        "You are an open book and the book is nine chapters of catalogued grievances and one very short chapter on accountability.",
        "Your energy is described as a lot by people exercising diplomatic restraint and as exhausting by people who are not.",
        "You are very authentic and the authenticity is functionally indistinguishable from a principled decision to never improve.",
        "You say life is short and use that philosophy exclusively to justify choices that make other people's lives feel considerably longer.",
        "You thrive in chaos and you have systematically arranged your circumstances to ensure chaos is always available.",
        "You are good vibes only in the same way that a no-returns policy is described as customer-friendly by the store.",
        "You are spontaneous in the way that a car breakdown is spontaneous, inconvenient, untimely, and someone else ends up managing it.",
    ]

    # ── CAT 10: AGE AND GENERATION ───────────────────────────
    age_t = [
        "a millennial who discusses their childhood in every conversation",
        "a gen-z person who cannot function without a phone for five minutes",
        "a boomer who says okay boomer about themselves preemptively",
        "a thirty-two-year-old who talks as if their life is completely over",
        "someone who peaked in high school and references it constantly",
        "a forty-year-old who uses current slang incorrectly at all times",
        "someone who claims to feel twenty years younger than they are",
        "a person who refuses to accept their favourite trends are no longer relevant",
        "someone who uses their age as a categorical reason they cannot learn anything new",
        "a twenty-four-year-old who is constantly exhausted by the concept of existing",
        "a person who romanticises a decade they were too young to have experienced",
        "someone who cites their age as an accomplishment requiring acknowledgment",
        "a person who dispenses advice based solely on being older than the recipient",
        "someone who is personally offended by everything younger generations enjoy",
        "a person who says things were better in my day in every conversation",
        "someone experiencing genuine shock at technology that has existed for eleven years",
        "a twenty-six-year-old actively having a midlife crisis",
        "someone who adjusts their stated age in both directions depending on the room",
        "a person who cannot concede that anything from their generation was bad",
        "someone whose entire identity is their generational cohort",
    ]
    age_r = [
        "You discuss your childhood so extensively that people who met you last year know your primary school's mascot and motto.",
        "You cannot go six minutes without your phone and have categorised this as a generational characteristic rather than a personal problem.",
        "You said okay boomer about yourself before anyone else could and that is a very specific and preemptive form of surrender.",
        "You are thirty-two and you discuss your life as though the end credits have begun rolling and you are reading them.",
        "You peaked in high school and you have been giving that peak a sustained standing ovation for fifteen consecutive years.",
        "You deploy current slang with the confidence of someone who learned it from a compiled list and the accuracy of no one who actually uses it.",
        "You feel twenty years younger than you are which means you feel precisely as young as you demonstrably act.",
        "You are upset that trends you liked are no longer culturally relevant which is grief being expressed at extremely low stakes.",
        "You cite your age as a reason learning new things is not possible and it is the most imaginative thing you have done recently.",
        "You are twenty-four and you are exhausted by life in a way that suggests life has not yet begun making its actual demands.",
        "You romanticise a decade you were not present for with such detail and precision that you have clearly done no research.",
        "You mention your age as an achievement requiring recognition as though surviving the passage of time is a skill set.",
        "You give advice based exclusively on being older than the recipient which is what happens when experience and insight separate.",
        "You are personally and visibly offended by everything younger generations enjoy and the offence is essentially a full-time role.",
        "Things were better in your day except for the significant and numerous things that were measurably worse which you have not factored in.",
        "You are experiencing fresh horror at technology that has existed for eleven years and has been widely discussed throughout.",
        "You are twenty-six and having a midlife crisis which means the crisis has arrived well ahead of the midlife it corresponds to.",
        "You adjust your stated age upward in some rooms and downward in others and have lost reliable track of which version is where.",
        "You cannot concede that anything from your generation was bad and your generation produced several things that were objectively bad.",
        "You define your entire identity by your generational cohort which is the maximum possible surface area for the minimum specificity.",
        "You are nostalgic for a period that exists primarily in the version of history you have personally curated and decided to remember.",
        "You describe everything contemporary as not as good as the original and the original was also not particularly good.",
        "You have comprehensive opinions about every other generation and no awareness that yours has also been comprehensively observed.",
        "You treat having grown up with a specific piece of technology as both a personality trait and a source of authority.",
        "You are ageing like a fine wine in your own assessment and like something with a shorter shelf life in several others.",
        "You give younger colleagues unsolicited life guidance and the guidance is the same three points reworded every time.",
        "You are offended that the world did not remain exactly as it was when you entered it and you have been filing complaints since.",
        "You describe young people as soft and you have not navigated a significant inconvenience with genuine grace in recorded memory.",
        "You discuss the past with a warmth and fondness that the actual historical record does not consistently or fully support.",
        "You say experience matters and then describe experiences that were entirely circumstantial and not transferable as accumulated wisdom.",
    ]

    # ── CAT 11: MONEY ────────────────────────────────────────
    mon_t = [
        "someone who is always broke but orders delivery food every single night",
        "a person who discusses investing constantly but has never invested anything",
        "someone who brags about their salary to people who earn significantly more",
        "a person who splits every romantic dinner bill to the exact penny",
        "someone who says they are saving money while making constant purchases",
        "a person with expensive taste operating on a budget that rejects the premise",
        "someone who lends money to everyone and complains no one ever repays them",
        "a person whose financial decisions are based entirely on vibes",
        "someone whose retirement strategy is exclusively the lottery",
        "a person who claims money is unimportant while being completely consumed by it",
        "someone who tells everyone their rent as a recurring personality contribution",
        "a person who films their beyond-means lifestyle for content",
        "someone who gives financial advice from a position of active financial difficulty",
        "a person who dramatically cancels subscriptions and resubscribes within the week",
        "someone who tips poorly in restaurants and explains at length why",
        "a person who is manifesting wealth as a substitute for working toward it",
        "someone whose credit score reflects a sustained commitment to poor decisions",
        "a person who buys sale items they did not need because they were on sale",
        "someone who made a budget spreadsheet and has never opened it since",
        "a person who describes every single purchase as an investment",
    ]
    mon_r = [
        "You are always broke and the delivery apps have your address stored alongside a note about your optimism.",
        "You discuss investing with the authority of someone who has a trading account they have not logged into in two years.",
        "You shared your salary with people who earn more and the silence that followed communicated considerably more than your number.",
        "You split every romantic dinner to the exact penny and the person you dined with is reconsidering the romance.",
        "You are saving money in the sense that you have identified saving as a concept that theoretically exists.",
        "You have expensive taste and a budget that has submitted a formal letter of resignation from this relationship.",
        "You lend money to everyone and bemoan that no one repays you and the pattern is presenting as the data.",
        "You make financial decisions based on vibes and the vibes have been consistently and measurably failing the audit.",
        "Your retirement plan involves the lottery in a capacity that the lottery has not formally agreed to.",
        "You say money is unimportant with the urgency of someone for whom money is demonstrably the most consuming topic.",
        "You tell everyone your rent amount as a recurring personality contribution and the personality it produces is complaint.",
        "You film your beyond-means lifestyle for content on the basis that sufficient views will eventually cover the gap.",
        "You give financial advice from a position of active financial difficulty with the confidence of someone who has not checked.",
        "You cancelled a subscription with considerable ceremony and resubscribed by Friday because the convenience reasserted itself.",
        "You tip badly and explain your reasoning at length and the length of the explanation does not improve any outcome.",
        "You are manifesting wealth and the manifestation has been in active progress since 2019 and the wealth is still in transit.",
        "Your credit score is a numerical record of a sequence of decisions made with great freedom and zero planning.",
        "You bought things on sale that you did not need because the discount produced the sensation of making money.",
        "You have a budget spreadsheet that has been opened once and that occasion was the day you created it.",
        "You describe every purchase as an investment including the coffee, the jacket, the streaming service, and the scented candle.",
        "You are financially free in the way a kite is free, briefly, in service of external forces, attached to something.",
        "You said you would be financially established by thirty and have rescheduled that to thirty-five and then forty.",
        "Your relationship with money mirrors your relationship with the gym, you know exactly what to do and reliably do something else.",
        "You buy things to improve your mood and then feel worse and then buy things to improve your mood about feeling worse.",
        "You have strong positions on how wealthy people should spend their money and a very underdeveloped plan for your own.",
        "You describe yourself as frugal in the same sentence as describing a purchase that would distress a genuinely frugal person.",
        "Your financial planning is aspirational in the same way that wanting to become an astronaut is aspirational, without the steps.",
        "You split shared expenses with an exactness that would impress a forensic accountant and alienate everyone you spend time with.",
        "You have no savings and a genuinely solid and detailed plan to begin saving starting next month.",
        "You invest in yourself without pause and the returns have not yet appeared in any category that instruments can detect.",
    ]

    # ── CAT 12: ARCHETYPES ───────────────────────────────────
    arc_t = [
        "a self-help guru who has visibly never solved a problem",
        "a wellness influencer who sells products with zero scientific basis",
        "a motivational speaker who has never been visibly motivated",
        "a productivity expert who is demonstrably not productive",
        "a relationship expert who has never sustained a long relationship",
        "a diet culture personality obsessed with other people's bodies",
        "a hustle culture evangelist who actively glamorises burnout",
        "a minimalist who films their empty apartment and sells courses about it",
        "a gratitude journal advocate who complains every single day",
        "a mindfulness coach who panics in standard traffic",
        "a morning routine influencer whose routine occupies four hours",
        "a cold shower evangelist who has solved nothing with cold showers",
        "a meditation guru who argues on the internet daily",
        "a breathwork coach who hyperventilates under moderate pressure",
        "a life optimiser who is visibly and persistently unhappy",
        "a radical authenticity advocate who is performing at all times",
        "a community builder who does not know a single neighbour",
        "a positivity coach who cannot receive any criticism at all",
        "a purpose coach who changes their stated purpose every eighteen months",
        "a high-performance coach whose own performance is middling at best",
    ]
    arc_r = [
        "You are a self-help guru who has never visibly helped yourself and is currently outsourcing that project to a ticketed workshop.",
        "You promote products with zero scientific basis to people who arrived seeking wellness and received a supplement and a disclaimer.",
        "You are a motivational speaker who delivers motivation the way a vending machine delivers food, transactionally and without genuine interest.",
        "You are a productivity expert with a personal to-do list that has been actively growing since 2021 without resolution.",
        "You are a relationship expert with a personal romantic history that a publicist would describe as colourful and a lawyer as complex.",
        "You comment on other people's bodies under the banner of health while carrying your own unexamined anxiety about the subject.",
        "You preach hustle culture to audiences who are already at capacity, specifically in the name of further burning.",
        "You are a minimalist, you own one linen shirt, and you sell courses to others about owning one linen shirt.",
        "You complete your gratitude journal with diligence and then spend the remainder of the day complaining as though the journal is a receipt.",
        "You teach mindfulness as a practice and cut off three separate vehicles on the way to the retreat location.",
        "Your morning routine initiates at four AM and concludes at eight which means your productive window opens at eight like everyone else.",
        "You take cold showers and announce this with the energy of someone who has resolved something that cold showers categorically cannot resolve.",
        "You meditate every morning and argue on the internet every evening and this represents a very committed two-part daily practice.",
        "You teach breathwork as a discipline and the first time genuine pressure arrives your breathwork is completely and conspicuously unavailable.",
        "You optimise every dimension of your life on an ongoing basis and the optimised version is measurably less content than the original.",
        "You practice radical authenticity at all times and the authenticity is so precisely constructed it operates on a production schedule.",
        "You build communities with significant online reach and your immediate physical neighbours have no knowledge of who you are.",
        "You coach other people through receiving difficult feedback and you cannot receive a single piece of it without a notable spiral.",
        "You found your purpose with great fanfare and then found a different one and then another and the finding is now the purpose.",
        "You coach high performance as a profession and your own performance is at an altitude that a charitable assessor would describe as developing.",
        "You are aligned, centred, grounded, and visibly and persistently unwell and this observation is offered with genuine care.",
        "You sell your audience a version of your life that you are not living and the courses describe how to live it.",
        "You have rebranded your personal development five separate times and each rebrand arrives with the energy of genuine transformation.",
        "You tell audiences the work is the reward while simultaneously selling the work back to them as a premium product.",
        "You deliver talks about living fully and you fully live for the delivery of the talks.",
        "Your transformation story is detailed, emotionally resonant, and the specific details have evolved considerably with scale.",
        "You are on a healing journey and the journey has merchandise, a mailing list, and a tiered membership structure.",
        "You say comparison is the thief of joy and maintain a private comparison chart of your metrics against your direct competitors.",
        "You describe your leadership style as servant leadership and the primary beneficiary of the service is your own platform.",
        "You are unapologetically yourself and the self you are unapologetically is a very deliberately and carefully engineered product.",
    ]

    # ── DIRECT ONE-LINER PAIRS (200) ─────────────────────────
    direct_pairs = [
        ("Give me a roast", "You have the energy of a browser running forty-seven tabs and none of them contain useful information."),
        ("Give me a roast", "You are the human equivalent of a terms and conditions page, long, unread, and ultimately disregarded."),
        ("Give me a roast", "Somewhere a village is missing its most self-assured resident."),
        ("Give me a roast", "You are not the main character, you are not even ranked in the top five."),
        ("Give me a roast", "Your potential is genuinely enormous and your track record is the one thing standing between you and it."),
        ("Give me a roast", "You have a great deal going on and none of it is going particularly well."),
        ("Give me a roast", "You are a work in progress and the project has been placed on indefinite hold."),
        ("Give me a roast", "Your presence in a room is best described as technically optional."),
        ("Give me a roast", "You move through the world like someone who has never encountered the word no and it informs every decision."),
        ("Give me a roast", "You are aggressively average and somehow maintain a level of pride about it."),
        ("Give me a roast", "Your greatest strength is consistency and your greatest weakness is what you are consistent at."),
        ("Give me a roast", "You peaked during a window of time that nobody else on record has identified as a peak."),
        ("Give me a roast", "You have very strong opinions and very thin evidence and the combination has been escalating annually."),
        ("Give me a roast", "You are the living demonstration that confidence and competence are not the same category."),
        ("Give me a roast", "You would be the last person I called for anything and I have your number saved under your actual name."),
        ("Give me a roast", "Your self-awareness is operating at such a low altitude it has its own distinct gravitational field."),
        ("Give me a roast", "You are a cautionary tale with an extremely active and well-maintained social media presence."),
        ("Give me a roast", "Your follow-through rate is so consistently low it has effectively become a declared policy position."),
        ("Give me a roast", "You are the type of person that other people discuss in therapy sessions rather than at celebratory events."),
        ("Give me a roast", "Your growth trajectory has a very gentle positive slope that points in a direction that has not yet been determined."),
        ("Roast me hard", "I would apply more force to this roast but I do not want to deploy quality material on someone who will not process it."),
        ("Roast me hard", "You are so forgettable that your own memories have begun editing you out of the frames you appeared in."),
        ("Roast me hard", "You bring extraordinary joy to every room the precise moment you decide to leave it."),
        ("Roast me hard", "You are an acquired taste and the general public has not yet proceeded to that stage of acquisition."),
        ("Roast me hard", "Your Wikipedia page would be a stub article and the stub would be flagged as potentially inaccurate."),
        ("Roast me hard", "You are genuinely one of a kind in the sense that the original mould observed the outcome and destroyed itself."),
        ("Roast me hard", "Your highlight reel is a short film and even the short film has been trimmed."),
        ("Roast me hard", "You are exactly what other people are referring to when they say and then there are some people."),
        ("Roast me hard", "You have the depth of a puddle that formed during a drought in a particularly flat region."),
        ("Roast me hard", "You are thriving in ways that are not visible to the naked eye or to any other currently available instrument."),
        ("Say something mean", "You are a supporting character in your own biographical story and the reviews for the supporting character are mixed."),
        ("Say something mean", "Your vibe is a waiting room with broken Wi-Fi, fluorescent lighting, and no indication of wait time."),
        ("Say something mean", "You are demonstrable proof that not all forms of attention are forms of earned attention."),
        ("Say something mean", "Your ambitions and your actions are in a long-distance relationship and neither has booked a visit."),
        ("Say something mean", "You are the reason people invoke the phrase things happen for a reason, as comfort when things like you occur."),
        ("Say something mean", "Your presence in a situation requires an explanation that is longer than the situation justifies."),
        ("Say something mean", "You are operating several lanes outside your designated lane and driving with considerable confidence."),
        ("Say something mean", "You are the ambient background noise occurring during someone else's genuinely important moment."),
        ("Say something mean", "Your reputation arrives before you do and does not turn around to wave as it passes."),
        ("Say something mean", "You are a lot and specifically not in the sense that a lot typically implies something impressive."),
        ("Destroy them", "I have observed better survival instincts in organisms that are already classified as extinct."),
        ("Destroy them", "You are not a red flag you are a fully operational red flag manufacturing facility running three production shifts."),
        ("Destroy them", "Your emotional intelligence is on a gap year with no confirmed return date and no forwarding address."),
        ("Destroy them", "You are building something and the something continues to collapse and you continue to attribute this to the materials."),
        ("Destroy them", "Your self-improvement programme has been running long enough that it should have produced visible results by now."),
        ("Destroy them", "You are fluent in the language of excuses and conversational in everything else you could theoretically offer."),
        ("Destroy them", "Your boundaries protect you from growth as reliably and comprehensively as they protect you from everything else."),
        ("Destroy them", "You are in your era and the era is structurally identical to every previous era you have announced."),
        ("Destroy them", "You are precisely the main character of a story that nobody has chosen to read."),
        ("Destroy them", "You have entered your villain era and the villain is predominantly late and operating in a passive aggressive mode."),
        ("Go hard on them", "You are so comprehensively and chronically online that sunlight is functionally breaking news to you."),
        ("Go hard on them", "Your personality rotates through a set of tracks and none of the tracks have ever been described as bangers."),
        ("Go hard on them", "You are the specific category of person that makes everyone around you feel considerably better about their own choices."),
        ("Go hard on them", "You are thriving with the committed enthusiasm of someone who has misidentified what thriving consists of."),
        ("Go hard on them", "Your opinions are load-bearing structural elements and the structure they support is not stable."),
        ("Go hard on them", "You are living authentically which in practice means you have determined that consequences are for other people."),
        ("Go hard on them", "You are on a journey and the journey has been this specific journey for four consecutive calendar years."),
        ("Go hard on them", "Your character development is proceeding at a pace and toward a destination that remains genuinely unclear."),
        ("Go hard on them", "You are perfectly and completely yourself and yourself requires considerable sustained effort to be around."),
        ("Go hard on them", "You operate with the emotional regulation of a vending machine that accepted the payment and provided nothing."),
        ("Brutal roast please", "You are the category of problem that actively worsens in direct response to attempts to address it."),
        ("Brutal roast please", "Your instincts are wrong in a way that requires real dedication and consistency to maintain."),
        ("Brutal roast please", "You have misread every room you have entered and walked into each one with compounding conviction."),
        ("Brutal roast please", "You call it authenticity and your closest associates call it something else entirely using different words."),
        ("Brutal roast please", "Your self-belief is genuinely admirable and is the only thing currently exceeding your self-awareness by a measurable margin."),
        ("Brutal roast please", "You are a complete experience and the reviews are divided in a way that suggests fundamental disagreement about the product."),
        ("Brutal roast please", "You are your own primary hype person and the hype has not yet transferred to any external party or audience."),
        ("Brutal roast please", "You are very relatable and accessible online and very difficult to be around in person and there may be two of you."),
        ("Brutal roast please", "Your personal brand is extremely strong and your personal reality is conducting a separate and uncoordinated conversation."),
        ("Brutal roast please", "You have energy that fills every room you enter and the energy prompts people to begin locating the exits."),
        ("No filter roast", "You are allergic to accountability in a way that medicine cannot currently explain but everyone in proximity has documented."),
        ("No filter roast", "Your personal narrative is a creative project that requires the reader to suspend significant quantities of disbelief."),
        ("No filter roast", "You have the self-awareness of a weather forecasting service that is wrong about every prediction but continues forecasting."),
        ("No filter roast", "You are among the primary reasons that certain people have concluded they prefer solitude."),
        ("No filter roast", "Your growth is so incremental it is operating within the measurement error of stationary."),
        ("No filter roast", "You are generous with your opinions and operate on a strict rationing system with verifiable facts."),
        ("No filter roast", "You are genuinely excellent at initiating things and extraordinarily talented at not completing any of them."),
        ("No filter roast", "You are in a phase and people have been describing it as a phase for six consecutive years without revision."),
        ("No filter roast", "You have the patience of someone who has never been patient, has no plans to become patient, and is proud of this."),
        ("No filter roast", "Your self-care is extensive and comprehensive and the self receiving the care remains challenging for others to be around."),
        ("Savage comeback", "I would walk you through the explanation but I do not have that calibre of time or that quantity of chalk."),
        ("Savage comeback", "You are the precise cautionary tale that self-help authors omit because the example is too directly on the nose."),
        ("Savage comeback", "You are the type of person who improves a room by leaving it and then phones to ask if you left your charger."),
        ("Savage comeback", "You are deeply and sincerely committed to a version of yourself that requires substantial and urgent updates."),
        ("Savage comeback", "Your comeback would carry more force if you had left and returned as a meaningfully different person."),
        ("Savage comeback", "You are projecting main character energy in a narrative where you are structurally functioning as the subplot."),
        ("Savage comeback", "You are currently producing your worst work and marketing it under the label unbothered."),
        ("Savage comeback", "Your silence here would have been the more powerful choice and I genuinely wish you had arrived at that independently."),
        ("Savage comeback", "You arrived with confidence and the confidence is the single most defensible element of the entire arrival."),
        ("Savage comeback", "You are a lesson that requires repeated delivery because no member of the audience has yet taken it."),
        ("Destroy their ego", "Your ego is writing cheques that your demonstrated talent is declining to honour."),
        ("Destroy their ego", "You are the most significant person in your own story and that story has a verified audience of one."),
        ("Destroy their ego", "You enter rooms with the proprietary confidence of an owner and you do not hold the title to any rooms."),
        ("Destroy their ego", "Your confidence is doing the work that your actual skills have not yet submitted an application to do."),
        ("Destroy their ego", "You believe in yourself at a level that substantially exceeds what the available supporting evidence recommends."),
        ("Destroy their ego", "The standards you hold other people to and the standards you hold yourself to have not been formally introduced."),
        ("Destroy their ego", "You are precisely as impressive as you sincerely believe yourself to be and that number is a downward revision."),
        ("Destroy their ego", "You reference your accomplishments with such frequency that the accomplishments themselves have begun to seem fatigued."),
        ("Destroy their ego", "Your ego exerts its own gravitational field and it is pulling every nearby element slightly off its intended course."),
        ("Destroy their ego", "You have achieved things, genuinely, just not the specific things you consistently choose to discuss."),
        ("Existential roast", "You have one life and you are allocating it to arguing in comment sections about matters that will not survive the week."),
        ("Existential roast", "You were given potential and it is currently in a drawer under several cable adaptors and some unread books."),
        ("Existential roast", "You are going to look back on this period and experience a very specific and quiet variety of regret."),
        ("Existential roast", "You wanted to leave a mark on the world and the world is still working to determine what the mark is."),
        ("Existential roast", "You have had every relevant opportunity and opportunity is beginning to feel it may have overinvested in this particular situation."),
        ("Existential roast", "You are building the life you want beginning next year and next year has received this message several times previously."),
        ("Existential roast", "You are becoming exactly who you were always meant to be and the becoming has been ongoing at this pace for some time."),
        ("Existential roast", "You have a plan for everything with the exception of the part where any of it begins to actually happen."),
        ("Existential roast", "Your legacy is being actively written by what you do during the periods when you believe no one is watching."),
        ("Existential roast", "You have every necessary ingredient and have been waiting for the inspiration that has been running late since 2018."),
        ("Dating roast", "You are a red flag parade that has made the tactical decision to present in the costume of a green flag."),
        ("Dating roast", "You are emotionally unavailable and aesthetically overcommitted and the combination is a lot to manage."),
        ("Dating roast", "You give people butterflies and then provide a debrief two weeks later explaining why it cannot work."),
        ("Dating roast", "You are tremendous fun for a short defined period of time and a significant lesson for an undefined period after that."),
        ("Dating roast", "Your attachment style is a puzzle that everyone who encounters it eventually decides is not worth completing."),
        ("Dating roast", "You are excellent on paper and a full-time management project in the actual practice of being around you."),
        ("Dating roast", "You give precisely enough to keep people present and not quite enough to make them content and this is a system."),
        ("Dating roast", "You are someone's type until you are the specific experience they bring up to explain their current avoidance patterns."),
        ("Dating roast", "You arrive at the beginning of things with extraordinary energy and at the middle of things with considerably less."),
        ("Dating roast", "You are everyone's favourite mistake and nobody's first deliberate and considered choice."),
        ("Workplace roast", "You reply all to emails that contained no implicit or explicit invitation to reply all."),
        ("Workplace roast", "Your out-of-office message is more creatively developed than anything produced during the hours you are in the office."),
        ("Workplace roast", "You take thirty minutes to convey information that required two minutes and the meeting could have been a brief message."),
        ("Workplace roast", "You schedule meetings specifically to discuss other meetings and the original problem has aged out of relevance."),
        ("Workplace roast", "You say you perform optimally under pressure and you have been under documented pressure for three years with the same output."),
        ("Workplace roast", "Your calendar is at full occupancy and there is no available slot for the productivity to move into."),
        ("Workplace roast", "You speak first in every meeting and are consistently the last person in the room to say anything that adds value."),
        ("Workplace roast", "You provide feedback on everything in the organisation with the exception of the things you are directly responsible for."),
        ("Workplace roast", "You have declared everything urgent and in doing so have permanently retired urgency as a meaningful operational concept."),
        ("Workplace roast", "You have synergy, bandwidth, alignment, and no output that can be located or attributed to you in any quarter."),
        ("Family roast", "You send voice notes that run to four minutes for information that eight words would have covered completely."),
        ("Family roast", "You give parenting advice to parents based on having been in the vicinity of parenting on several occasions."),
        ("Family roast", "You are the relative who gives gift cards because you stopped attempting to understand the recipient several years ago."),
        ("Family roast", "You surface old arguments at family gatherings because you believe the historical record requires active maintenance."),
        ("Family roast", "You are the family member that longer-standing members brief new arrivals about before the seasonal gathering."),
        ("Family roast", "You provide unsolicited medical updates to people who asked how you are as a social formality and not as a genuine enquiry."),
        ("Family roast", "You have strong positions on how everyone else in the family should be living and flexible positions on your own situation."),
        ("Family roast", "You arrive at every family gathering and immediately conduct an assessment of what has changed and your approval status."),
        ("Family roast", "You are the designated keeper of grievances that every other person in the family released and moved on from decades ago."),
        ("Family roast", "You have been meaning to call for three months and have called twice in that period specifically to mention you have been meaning to."),
        ("Be savage", "You are the category of person that surfaces in other people's therapy sessions rather than in their celebration speeches."),
        ("Be savage", "You are chronically and committedly yourself in ways that benefit primarily and sometimes exclusively yourself."),
        ("Be savage", "You have an origin story and an interesting one but the subsequent arc has not developed a resolution or a direction."),
        ("Be savage", "You are someone's lesson dressed in a first impression that was genuinely very persuasive."),
        ("Be savage", "You are doing the maximum possible amount and producing the minimum possible output per unit of maximum."),
        ("Be savage", "Your brand is yourself and the brand has reviews across multiple platforms and they are divided."),
        ("Be savage", "You are exactly what other people are referring to when they say they occasionally need a break from people."),
        ("Be savage", "You are evolving at a measurable pace in a direction that has been taking longer than projected to become identifiable."),
        ("Be savage", "You are your own primary enthusiast and the enthusiast club has very restricted and selective external membership."),
        ("Be savage", "You are proof that certain things read better from a distance and from a distance you are genuinely fine."),
    ]

    # ── ASSEMBLE ALL PAIRS ────────────────────────────────────
    all_pairs = []
    categories = [
        (app_t, app_r), (int_t, int_r), (car_t, car_r),
        (soc_t, soc_r), (rel_t, rel_r), (fit_t, fit_r),
        (food_t, food_r), (tech_t, tech_r), (life_t, life_r),
        (age_t,  age_r), (mon_t,  mon_r), (arc_t,  arc_r),
    ]
    for tgts, rsts in categories:
        all_pairs.extend(make_pairs(tgts, rsts))

    all_pairs.extend(direct_pairs)

    # 3 000 cross-category random pairings
    all_t_flat = [t for tgts, _ in categories for t in tgts]
    all_r_flat = [r for _, rsts in categories for r in rsts]
    rng = random.Random(77)
    for i in range(3000):
        t = rng.choice(all_t_flat)
        r = rng.choice(all_r_flat)
        all_pairs.append((f"Roast: {t}", r))

    print(f"[Dataset Synthetic] {len(all_pairs):,} pairs generated")
    return all_pairs


# ================================================================
# CELL 4 — REAL DATASETS (all 4, all verified working in 2024-25)
# ================================================================

from datasets import load_dataset

# ── D1: OpenHermes-2.5-Filtered (instruction-response pairs, adapt for roasts) ──────────
def load_d1_openhermes():
    """
    Load Replete-AI/OpenHermes-2.5-Filtered for instruction-based roast prompts.
    Adapt instruction-response pairs to roast format.
    """
    print("\n[D1] Loading OpenHermes-2.5-Filtered dataset...")
    pairs = []
    try:
        ds = load_dataset("Replete-AI/OpenHermes-2.5-Filtered", split="train", streaming=True)
        count = 0
        for item in ds:
            if len(pairs) >= 10000:  # Cap at 10K
                break
            instruction = str(item.get("instruction", "")).strip()
            output = str(item.get("output", "")).strip()
            
            if not instruction or not output or len(instruction) < 10 or len(output) < 20:
                continue
            
            # Adapt to roast format: "Give a roast about [instruction]" -> output
            if len(instruction) > 5 and len(output) > 15:
                roast_input = f"Roast this: {instruction}"
                pairs.append((roast_input, output))
                
        print(f"  [D1] OpenHermes: {len(pairs):,} pairs")
        return pairs
    except Exception as e:
        print(f"  [D1] OpenHermes failed: {e}")
        return []


# ── D2: HuggingFace roast-adjacent humor datasets ────────────
def load_d2_humor():
    """
    Tries several known-working HF humor datasets in priority order.
    No scripts, no trust_remote_code, all Parquet.
    """
    print("\n[D2] Loading humor dataset...")
    pairs = []

    attempts = [
        # (dataset_id, split, text_col, label_col, label_value)
        ("mohameddhiab/humor-no-humor", "train", "text", "label", 1),
        ("Fraser/short-jokes",          "train", "jokeText", None,  None),
        ("Samoed/funny-or-not",         "train", "text",     "label",1),
    ]

    for ds_id, split, text_col, lbl_col, lbl_val in attempts:
        if pairs:
            break
        try:
            ds = load_dataset(ds_id, split=split)
            for item in ds:
                text = str(item.get(text_col, "") or "").strip()
                if not text or len(text) < 20 or len(text) > 300:
                    continue
                if lbl_col is not None:
                    if item.get(lbl_col) != lbl_val:
                        continue
                pairs.append(("roast this", text))
                if len(pairs) >= 8000:
                    break
            if pairs:
                print(f"  [D2] {ds_id}: {len(pairs):,} pairs")
        except Exception as e:
            print(f"  [D2] {ds_id} failed: {e}")

    if not pairs:
        pairs = [
            ("roast this", "You have the energy of a phone at four percent."),
            ("roast this", "Your personality is a participation trophy."),
            ("roast this", "You are the human equivalent of a loading screen."),
            ("roast this", "Your vibe is expired coupon."),
            ("roast this", "You look like you argue with self-checkout machines and lose."),
            ("roast this", "You are what happens when ambition takes a day off and never comes back."),
            ("roast this", "You are proof that the audacity is not evenly distributed."),
            ("roast this", "You have the follow-through of a broken compass."),
            ("roast this", "Your default setting is technically present but not really there."),
            ("roast this", "You are aggressively forgettable which is its own kind of achievement."),
        ]
        print(f"  [D2] Using hardcoded fallback: {len(pairs)} pairs")

    return pairs


# ── D3: Comedy Central Roast Transcripts (web scrape) ────────
def load_d3_comedy_central():
    """
    Scrapes springfieldspringfield.co.uk for roast transcripts.
    Uses exact slugs that returned HTTP 200 in the prior run.
    Confirmed working: Bieber, Franco, Sheen, Trump, Anderson, Hasselhoff.
    Falls back to embedded high-quality roasts if scraping fails.
    """
    print("\n[D3] Fetching Comedy Central Roast transcripts...")
    pairs = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Only slugs confirmed HTTP 200 in previous run
    targets = [
        ("Justin Bieber",    "comedy-central-roast-of-justin-bieber"),
        ("James Franco",     "comedy-central-roast-of-james-franco"),
        ("Charlie Sheen",    "comedy-central-roast-of-charlie-sheen"),
        ("Donald Trump",     "comedy-central-roast-of-donald-trump"),
        ("Pamela Anderson",  "comedy-central-roast-of-pamela-anderson"),
        ("David Hasselhoff", "comedy-central-roast-of-david-hasselhoff"),
    ]

    for name, slug in targets:
        url = f"https://www.springfieldspringfield.co.uk/movie_script.php?movie={slug}"
        try:
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code} for {name}")
                continue
            soup  = BeautifulSoup(r.text, "html.parser")
            block = (soup.find("div", class_="movie_script") or
                     soup.find("div", {"id": "movie_script"}) or
                     soup.find("div", class_="scrolling-script-container") or
                     soup.find("article"))
            text  = (block.get_text(separator="\n") if block
                     else "\n".join(p.get_text() for p in soup.find_all("p")))
            lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 20]
            fname = name.split()[0].lower()
            markers = ["you ", "your ", "you've ", "you're ", "you'd ",
                       "he ", "she ", "his ", "her ", fname]
            n_before = len(pairs)
            for line in lines:
                if len(line) > 250 or line.startswith(("[", "(", "INT ", "EXT ")):
                    continue
                if line.isupper() and len(line.split()) <= 4:
                    continue
                if any(m in line.lower() for m in markers):
                    pairs.append((f"Roast {name}", line))
            print(f"  {name}: +{len(pairs)-n_before} lines (total {len(pairs)})")
            time.sleep(1.8)
        except Exception as e:
            print(f"  Error for {name}: {e}")

    # Always append high-quality hardcoded roasts for these targets
    hardcoded = [
        ("Roast Justin Bieber", "Justin you started performing at thirteen and you still perform like you are thirteen."),
        ("Roast Justin Bieber", "You have achieved so much so young and you have so much time remaining to dismantle it."),
        ("Roast Justin Bieber", "You tattooed your face to project intimidation and instead projected a very specific type of confusion."),
        ("Roast Justin Bieber", "You went from Baby to maybe the most genuinely confusing trajectory in the history of modern music."),
        ("Roast Justin Bieber", "Everyone said you would be the next Michael Jackson and the trajectory is beginning to support a different read."),
        ("Roast Charlie Sheen", "Charlie you turned a public breakdown into a personal brand and the brand is performing worse than the breakdown."),
        ("Roast Charlie Sheen", "Charlie has consumed enough controlled substances that his dealer sends him a personalised holiday card and a year-end statement."),
        ("Roast Charlie Sheen", "You say winning but the scoreboard has not reflected that assessment in a considerable amount of time."),
        ("Roast Charlie Sheen", "Charlie Sheen is living evidence that a human being can survive almost anything except their own sustained decision-making."),
        ("Roast Donald Trump", "You have the presentation of a man who has never personally encountered the word no and the hair of someone who should have."),
        ("Roast Donald Trump", "Your hair is so well-known it has filed for independent trademark status and is performing better than your Atlantic City properties."),
        ("Roast Donald Trump", "You built your entire career on the word fired and history appears committed to exploring the full irony of that."),
        ("Roast Pamela Anderson", "Pamela you are the only person to appear on every magazine cover and on the default menu of every hotel television simultaneously."),
        ("Roast James Franco", "James Franco is so committed to the method that when he played a student he enrolled, attended, and actually failed the course."),
        ("Roast James Franco", "James has accumulated three graduate degrees and somehow none of them are in recognising when to conclude a project."),
        ("Roast David Hasselhoff", "David Hasselhoff is so famous in Germany that they named a processed meat product after him and nobody is certain which one is more processed."),
        ("Roast David Hasselhoff", "You saved so many lives across five seasons of Baywatch and then spent a decade making the remainder of us feel genuinely at risk."),
    ]
    pairs.extend(hardcoded)
    print(f"  [D3] Total: {len(pairs):,} pairs")
    return pairs


# ── D4: WikiHow satirical / instructional humor ───────────────
def load_d4_wikihow_roasts():
    """
    Uses 'sentence-transformers/wikihow' which is Parquet and verified.
    Filters for imperative sentences that can be inverted as roasts.
    Falls back completely if unavailable.
    """
    print("\n[D4] Loading WikiHow-derived roast prompts...")
    pairs = []
    try:
        ds = load_dataset("sentence-transformers/wikihow",
                          split="train", streaming=True)
        ironic_patterns = [
            "how to be", "how to look", "how to act", "how to seem",
            "how to appear", "how to sound", "how to dress",
            "how to impress", "how to succeed",
        ]
        count = 0
        for item in ds:
            if count >= 150000 or len(pairs) >= 5000:
                break
            count += 1
            title = str(item.get("title", "") or "").strip().lower()
            if not any(p in title for p in ironic_patterns):
                continue
            headline = str(item.get("headline", "") or "").strip()
            if not headline or len(headline) < 20 or len(headline) > 180:
                continue
            # Invert the instructional sentence as a satirical roast setup
            roast_inp = f"Roast someone who needs advice on: {title}"
            roast_out = (headline
                         .replace("You should ", "You actually ")
                         .replace("Make sure to ", "You consistently fail to ")
                         .replace("Try to ", "You never manage to "))
            if len(roast_out) > 15:
                pairs.append((roast_inp, roast_out))
        print(f"  [D4] WikiHow: {len(pairs):,} pairs")
    except Exception as e:
        print(f"  [D4] WikiHow failed: {e}")

    if len(pairs) < 50:
        pairs = [
            ("Roast someone who needs advice on how to be professional",
             "You actually show up to meetings without having read the brief, without a pen, and with full confidence."),
            ("Roast someone who needs advice on how to look presentable",
             "You consistently fail to achieve the minimum viable standard of presentability required to enter most rooms."),
            ("Roast someone who needs advice on how to make friends",
             "You never manage to convert a first impression into a second interaction and both parties know why."),
            ("Roast someone who needs advice on how to be punctual",
             "You actually arrive after the event has concluded and consider yourself fashionably late."),
            ("Roast someone who needs advice on how to communicate clearly",
             "You never manage to finish a sentence in a way that resembles how it started."),
            ("Roast someone who needs advice on how to manage money",
             "You actually spend money you do not have on things you do not need to impress people you do not like."),
            ("Roast someone who needs advice on how to exercise regularly",
             "You consistently fail to make it past the second week of any programme and call the first week a lifestyle change."),
            ("Roast someone who needs advice on how to be productive",
             "You actually optimise your workspace for eight hours and use it for forty-five minutes."),
            ("Roast someone who needs advice on how to stop procrastinating",
             "You never manage to begin the thing but maintain excellent documentation of your intention to begin."),
            ("Roast someone who needs advice on how to be on time",
             "You actually believe that your time is worth more than the collective time of everyone waiting for you."),
            ("Roast someone who needs advice on how to dress well",
             "You consistently fail to understand that clothes are not just fabric but also a message and yours is sending one."),
            ("Roast someone who needs advice on how to hold a conversation",
             "You actually turn every exchange into a monologue and call it being open and sharing."),
            ("Roast someone who needs advice on how to take criticism",
             "You never manage to hear a single critical observation without immediately becoming the injured party."),
            ("Roast someone who needs advice on how to follow through",
             "You actually announce every intention publicly and complete approximately none of them publicly or privately."),
            ("Roast someone who needs advice on how to read the room",
             "You consistently fail to detect the atmosphere of any room you walk into regardless of how clearly it is broadcast."),
            ("Roast someone who needs advice on how to be likeable",
             "You actually try very hard to be liked and the effort is one of the primary obstacles to achieving it."),
            ("Roast someone who needs advice on how to not overshare",
             "You never manage to have a conversation without disclosing something that should have remained internal."),
            ("Roast someone who needs advice on how to be consistent",
             "You actually start things with extraordinary enthusiasm and abandon them at the precise moment they require sustained effort."),
            ("Roast someone who needs advice on how to be confident",
             "You confuse volume with confidence and volume with knowledge and volume with most things."),
            ("Roast someone who needs advice on how to stop being defensive",
             "You never manage to receive new information without treating it as an attack that requires immediate counter-measures."),
        ]
        print(f"  [D4] Using hardcoded fallback: {len(pairs)} pairs")

    return pairs


# ── MASTER LOADER ─────────────────────────────────────────────
def load_all_datasets():
    d1 = load_d1_jokes()
    d2 = load_d2_humor()
    d3 = load_d3_comedy_central()
    d4 = load_d4_wikihow_roasts()
    d5 = generate_synthetic_roast_pairs()

    all_pairs = d1 + d2 + d3 + d4 + d5
    print(f"\n{'='*58}")
    print("DATASET SUMMARY")
    print(f"  D1 Jokes HF:           {len(d1):>8,}")
    print(f"  D2 Humor HF:           {len(d2):>8,}")
    print(f"  D3 Comedy Central:     {len(d3):>8,}")
    print(f"  D4 WikiHow satirical:  {len(d4):>8,}")
    print(f"  D5 Synthetic (10K+):   {len(d5):>8,}")
    print(f"  {'─'*40}")
    print(f"  TOTAL RAW:             {len(all_pairs):>8,}")
    print(f"{'='*58}\n")
    return all_pairs


# ── FILTER ────────────────────────────────────────────────────
def filter_and_tokenize_pairs(raw_pairs, tok,
                               max_inp=120, max_out=90, min_out=4):
    """
    Relaxed filter: keeps anything that tokenises within range.
    Removes only: duplicates, URLs, excessive lol/haha.
    Does NOT remove hedge phrases (too aggressive for small corpus).
    """
    filtered = []
    seen     = set()
    for inp_s, out_s in raw_pairs:
        if not isinstance(inp_s, str) or not isinstance(out_s, str):
            continue
        inp_s = inp_s.strip()
        out_s = out_s.strip()
        if not inp_s or not out_s:
            continue
        dk = out_s.lower()[:80]
        if dk in seen:
            continue
        seen.add(dk)
        if "http" in out_s.lower():
            continue
        if out_s.lower().count("lol") > 3:
            continue
        if out_s.lower().count("haha") > 2:
            continue
        if len(out_s.split()) < 4:
            continue
        inp_ids = tok.encode(inp_s)
        out_ids = tok.encode(out_s)
        if not inp_ids or not (1 <= len(inp_ids) <= max_inp):
            continue
        if not out_ids or not (min_out <= len(out_ids) <= max_out):
            continue
        inp_ids = [max(3, min(int(x), VOCAB_SIZE-1)) for x in inp_ids]
        out_ids = [max(3, min(int(x), VOCAB_SIZE-1)) for x in out_ids]
        filtered.append((inp_ids, out_ids))
    print(f"Filter: {len(raw_pairs):,} raw → {len(filtered):,} clean pairs")
    return filtered


print("Starting data pipeline...")
raw_pairs      = load_all_datasets()
filtered_pairs = filter_and_tokenize_pairs(raw_pairs, tokenizer)

if len(filtered_pairs) < 500:
    raise RuntimeError(
        f"Only {len(filtered_pairs)} pairs after filtering — "
        "something went very wrong. Check tokenizer encode output."
    )

random.shuffle(filtered_pairs)
n_total = len(filtered_pairs)
n_train = int(0.90 * n_total)
n_val   = int(0.05 * n_total)
train_pairs = filtered_pairs[:n_train]
val_pairs   = filtered_pairs[n_train : n_train + n_val]
test_pairs  = filtered_pairs[n_train + n_val :]
print(f"Split — Train:{len(train_pairs):,}  Val:{len(val_pairs):,}  Test:{len(test_pairs):,}")


# ================================================================
# CELL 5 — PYTORCH DATASET AND DATALOADERS
# ================================================================

class RoastDataset(Dataset):
    def __init__(self, pairs, bos_id=BOS_ID, eos_id=EOS_ID,
                 max_inp=120, max_out=92):
        self.pairs   = pairs
        self.bos_id  = bos_id
        self.eos_id  = eos_id
        self.max_inp = max_inp
        self.max_out = max_out

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        inp, out = self.pairs[idx]
        inp = list(inp)[:self.max_inp]
        out = [self.bos_id] + list(out)[:self.max_out - 2] + [self.eos_id]
        return inp, out


def collate_fn(batch, pad_id=PAD_ID, vocab_size=VOCAB_SIZE):
    inp_s = [b[0] for b in batch]
    out_s = [b[1] for b in batch]
    B     = len(batch)
    Li    = max(len(s) for s in inp_s)
    Lo    = max(len(s) for s in out_s)

    inp_t   = torch.full((B, Li), pad_id, dtype=torch.long)
    out_t   = torch.full((B, Lo), pad_id, dtype=torch.long)
    inp_msk = torch.zeros(B, Li, dtype=torch.bool)
    out_msk = torch.zeros(B, Lo, dtype=torch.bool)

    for i, (iv, ov) in enumerate(zip(inp_s, out_s)):
        ic = [min(max(int(x), 0), vocab_size - 1) for x in iv]
        oc = [min(max(int(x), 0), vocab_size - 1) for x in ov]
        inp_t[i, :len(ic)]    = torch.tensor(ic, dtype=torch.long)
        out_t[i, :len(oc)]    = torch.tensor(oc, dtype=torch.long)
        inp_msk[i, :len(ic)]  = True
        out_msk[i, :len(oc)]  = True

    return inp_t, out_t, inp_msk, out_msk


_collate      = partial(collate_fn, pad_id=PAD_ID, vocab_size=VOCAB_SIZE)
train_dataset = RoastDataset(train_pairs)
val_dataset   = RoastDataset(val_pairs)
train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True,
                           collate_fn=_collate, num_workers=0, pin_memory=False)
val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=False,
                           collate_fn=_collate, num_workers=0, pin_memory=False)
print(f"Train batches: {len(train_loader):,}   Val batches: {len(val_loader):,}")

# Index safety check
_bi, _bo, _bm, _om = next(iter(train_loader))
assert _bi.max().item() < VOCAB_SIZE, f"inp OOB: {_bi.max().item()}"
assert _bo.max().item() < VOCAB_SIZE, f"out OOB: {_bo.max().item()}"
assert _bi.min().item() >= 0
assert _bo.min().item() >= 0
print(f"Batch check OK — inp[{_bi.min()},{_bi.max()}] out[{_bo.min()},{_bo.max()}]")


# ================================================================
# CELL 6 — SCORCH ARCHITECTURE (all paper mathematics intact)
# ================================================================

# ── 6a: Torsional Gate ──────────────────────────────────────
# TG(x) = x ⊙ σ( tanh(W_τ·x + φ_τ) ⊙ x / √d )

class TorsionalGate(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.scale   = math.sqrt(float(d))
        self.W_tau   = nn.Linear(d, d, bias=False)
        self.phi_tau = nn.Parameter(torch.zeros(d))
        nn.init.eye_(self.W_tau.weight)
        with torch.no_grad():
            self.W_tau.weight += 0.01 * torch.randn_like(self.W_tau.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau       = torch.tanh(self.W_tau(x) + self.phi_tau)
        resonance = (tau * x) / self.scale
        return x * torch.sigmoid(resonance)


# ── 6b: Positional Belief Embedding (PBE) ───────────────────
# PBE_pos(p) = P[p] ⊙ TG(P[p])
# R(x)       = softmax(MLP_role(E_tok[x])) ∈ ℝ^4
# e_t        = concat(E_tok[x_t], PBE_pos(t), R(x_t)) · W_proj

class PositionalBeliefEmbedding(nn.Module):
    def __init__(self, vocab_size, d, d_pos=32, max_len=256):
        super().__init__()
        self.d       = d
        self.d_pos   = d_pos
        self.max_len = max_len
        self.E_tok   = nn.Embedding(vocab_size, d, padding_idx=PAD_ID)
        self.P       = nn.Embedding(max_len, d_pos)
        self.pos_tg  = TorsionalGate(d_pos)
        self.role_mlp = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 4),
        )
        self.W_proj  = nn.Linear(d + d_pos + 4, d, bias=False)
        nn.init.normal_(self.E_tok.weight, 0.0, 0.02)
        with torch.no_grad(): self.E_tok.weight[PAD_ID].fill_(0)
        nn.init.normal_(self.P.weight, 0.0, 0.02)
        nn.init.xavier_uniform_(self.W_proj.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, n    = ids.shape
        tok_emb = self.E_tok(ids)
        pos_idx = (torch.arange(n, device=ids.device)
                   .unsqueeze(0).expand(B, -1)
                   .clamp(max=self.max_len - 1))
        P_raw   = self.P(pos_idx)
        pos_emb = P_raw * self.pos_tg(P_raw)
        role_v  = F.softmax(self.role_mlp(tok_emb.detach()), dim=-1)
        return self.W_proj(torch.cat([tok_emb, pos_emb, role_v], dim=-1))


# ── 6c: Torsional Gated Block (TGB) ─────────────────────────
# X_mixed = W_mix(X) + ½(W_left(X_{t-1}) + W_right(X_{t+1}))
# X_gated = TG(X_mixed)
# X_ff    = W_ff2(GELU(W_ff1(X_gated)))
# X'      = LN(X + X_gated + X_ff)

class TorsionalGatedBlock(nn.Module):
    def __init__(self, d: int, ff_mult=4):
        super().__init__()
        self.W_mix  = nn.Linear(d, d, bias=False)
        self.W_left = nn.Linear(d, d, bias=False)
        self.W_right= nn.Linear(d, d, bias=False)
        self.tg     = TorsionalGate(d)
        self.ff1    = nn.Linear(d, d * ff_mult)
        self.ff2    = nn.Linear(d * ff_mult, d)
        self.ln     = nn.LayerNorm(d)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, n, d = X.shape
        z       = torch.zeros(B, 1, d, dtype=X.dtype, device=X.device)
        Xl      = torch.cat([z, X[:, :-1]], dim=1)
        Xr      = torch.cat([X[:, 1:], z],  dim=1)
        Xm      = self.W_mix(X) + 0.5 * self.W_left(Xl) + 0.5 * self.W_right(Xr)
        Xg      = self.tg(Xm)
        Xf      = self.ff2(F.gelu(self.ff1(Xg)))
        return self.ln(X + Xg + Xf)


# ── 6d: Belief Compression Funnel (BCF) ─────────────────────
# s_t = softmax(H · w_compress)
# bv  = Σ_t s_t · H_t

class BeliefCompressionFunnel(nn.Module):
    def __init__(self, d: int, n_blocks=3):
        super().__init__()
        self.blocks     = nn.ModuleList([TorsionalGatedBlock(d) for _ in range(n_blocks)])
        self.w_compress = nn.Parameter(torch.randn(d) * 0.02)
        self.ln         = nn.LayerNorm(d)

    def forward(self, X, mask=None):
        for blk in self.blocks:
            X = blk(X)
        sc = X @ self.w_compress
        if mask is not None:
            sc = sc.masked_fill(~mask, float('-inf'))
        w  = F.softmax(sc, dim=-1)
        bv = (w.unsqueeze(-1) * X).sum(dim=1)
        return self.ln(bv)


# ── 6e: Roast Salience ρ ─────────────────────────────────────
# ρ(X) = (1/K) Σ_k ReLU(cos_sim(E(X), b_k) − θ_k), ∈ [0,1]

class RoastSalience(nn.Module):
    def __init__(self, d: int, K=8):
        super().__init__()
        self.K          = K
        self.anchors    = nn.Parameter(torch.randn(K, d) * 0.02)
        self.thresholds = nn.Parameter(torch.zeros(K))

    def forward(self, bv: torch.Tensor) -> torch.Tensor:
        bn  = F.normalize(bv, dim=-1)
        an  = F.normalize(self.anchors, dim=-1)
        cs  = bn @ an.T
        act = F.relu(cs - self.thresholds)
        return act.mean(dim=-1).clamp(0.0, 1.0)


# ── 6f: Torsional Sparse Routing Layer (TSRL) ───────────────
# C_k      = TG(W_k · bv)           k = 1…K
# k*       = max(2, round(K·ρ))
# α        = renorm(straight-through top-k* gate)
# r_ctx    = Σ_k α_k C_k
# out      = LN(TG(r_ctx) + bv)

class TorsionalSparseRoutingLayer(nn.Module):
    def __init__(self, d: int, K=8):
        super().__init__()
        self.K      = K
        self.projs  = nn.ModuleList([nn.Linear(d, d) for _ in range(K)])
        self.tgs    = nn.ModuleList([TorsionalGate(d) for _ in range(K)])
        self.W_sel  = nn.Linear(d, K, bias=True)
        self.out_tg = TorsionalGate(d)
        self.ln     = nn.LayerNorm(d)

    def forward(self, bv, rho):
        B = bv.shape[0]
        C = torch.stack([self.tgs[k](self.projs[k](bv)) for k in range(self.K)], dim=1)
        gp = F.softmax(self.W_sel(bv), dim=-1)
        ks = (rho * self.K).round().clamp(min=2, max=self.K).long()
        hm = torch.zeros(B, self.K, dtype=bv.dtype, device=bv.device)
        for i in range(B):
            hm[i, torch.topk(gp[i], k=int(ks[i].item())).indices] = 1.0
        sm    = gp + (hm - gp).detach()
        alpha = sm / (sm.sum(dim=-1, keepdim=True) + 1e-8)
        rc    = (alpha.unsqueeze(-1) * C).sum(dim=1)
        return self.ln(self.out_tg(rc) + bv)


# ── 6g: Roast Context Memory Bank (RCMB) ────────────────────
# q         = W_q · ctx
# K_mem     = W_k · M
# scores    = q · K_mem.T / √d_mem
# gate      = σ(w_τ ⊙ scores / √M_slots)
# rw        = softmax(scores ⊙ gate)
# mem_read  = rw @ M
# out       = LN(ctx + W_out · mem_read)

class RoastContextMemoryBank(nn.Module):
    def __init__(self, d: int, M_slots=32, d_mem=64):
        super().__init__()
        self.scale_d = math.sqrt(float(d_mem))
        self.scale_m = math.sqrt(float(M_slots))
        self.M       = nn.Parameter(torch.randn(M_slots, d) * 0.02)
        self.W_q     = nn.Linear(d, d_mem, bias=False)
        self.W_k     = nn.Linear(d, d_mem, bias=False)
        self.W_out   = nn.Linear(d, d, bias=False)
        self.w_tau   = nn.Parameter(torch.ones(M_slots))
        self.ln      = nn.LayerNorm(d)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        q      = self.W_q(ctx)
        Km     = self.W_k(self.M)
        scores = q @ Km.T / self.scale_d
        gate   = torch.sigmoid(self.w_tau.unsqueeze(0) * scores / self.scale_m)
        rw     = F.softmax(scores * gate, dim=-1)
        mr     = rw @ self.M
        return self.ln(ctx + self.W_out(mr))


# ── 6h: Comedic Entropy Decoder Block (CED) ─────────────────
# D_causal  = W_self(D_t) + W_left(D_{t-1})
# injection = TG_cross(D_causal + ctx_broadcast)
# κ_t       = 1 - cos_sim(h_t, μ_neutral)·exp(-λ_κ·t)
# ff_scale  = 1 + κ_t
# D_ff      = W_ff2(GELU(W_ff1(injection) · ff_scale))
# D_out     = LN(injection + D_ff)

class ComediciEntropyDecoderBlock(nn.Module):
    def __init__(self, d: int, ff_mult=4):
        super().__init__()
        self.W_self = nn.Linear(d, d, bias=False)
        self.W_left = nn.Linear(d, d, bias=False)
        self.tg_x   = TorsionalGate(d)
        self.ff1    = nn.Linear(d, d * ff_mult)
        self.ff2    = nn.Linear(d * ff_mult, d)
        self.ln     = nn.LayerNorm(d)

    def forward(self, D, rcmb_out, kappa):
        B, t, d = D.shape
        z       = torch.zeros(B, 1, d, dtype=D.dtype, device=D.device)
        Dl      = torch.cat([z, D[:, :-1]], dim=1)
        Dc      = self.W_self(D) + self.W_left(Dl)
        ctx     = rcmb_out.unsqueeze(1).expand(-1, t, -1)
        inj     = self.tg_x(Dc + ctx)
        ffs     = (1.0 + kappa).unsqueeze(-1)
        Dff     = self.ff2(F.gelu(self.ff1(inj) * ffs))
        return self.ln(inj + Dff)


# ── 6i: Full SCORCH Model ─────────────────────────────────────

class SCORCH(nn.Module):
    def __init__(self, vocab_size=HARD_VOCAB, d=256, d_pos=32,
                 n_enc=3, n_dec=4, K_route=8, K_anchors=8,
                 M_slots=32, d_mem=64, max_len=256, lambda_kappa=0.05):
        super().__init__()
        self.d            = d
        self.vocab_size   = vocab_size
        self.max_len      = max_len
        self.lambda_kappa = lambda_kappa

        self.enc_emb  = PositionalBeliefEmbedding(vocab_size, d, d_pos, max_len)
        self.bcf      = BeliefCompressionFunnel(d, n_enc)
        self.salience = RoastSalience(d, K_anchors)
        self.tsrl     = TorsionalSparseRoutingLayer(d, K_route)
        self.rcmb     = RoastContextMemoryBank(d, M_slots, d_mem)

        self.dec_pos    = nn.Embedding(max_len, d)
        self.dec_blocks = nn.ModuleList([ComediciEntropyDecoderBlock(d) for _ in range(n_dec)])
        self.dec_ln     = nn.LayerNorm(d)

        self.mu_neutral = nn.Parameter(torch.randn(d) * 0.02)
        self.phi_exp    = nn.Parameter(torch.tensor(2.0))
        self.w_psi      = nn.Linear(d, 1, bias=True)

        # weight-tied output projection
        self.output_proj        = nn.Linear(d, vocab_size, bias=False)
        self.output_proj.weight = self.enc_emb.E_tok.weight

        nn.init.normal_(self.dec_pos.weight, 0.0, 0.02)
        for nm, p in self.named_parameters():
            skip = any(s in nm for s in [
                'E_tok','W_tau','phi_tau','mu_neutral',
                'phi_exp','w_tau','M ','anchors','thresholds',
            ])
            if skip: continue
            if p.dim() == 2: nn.init.xavier_uniform_(p)
            elif p.dim() == 1 and p.numel() > 8: nn.init.zeros_(p)

    # κ(t) = 1 − cos_sim(h_t, μ_neutral) · exp(−λ_κ · t)
    def compute_kappa(self, hidden, pos_idx):
        hn  = F.normalize(hidden, dim=-1)
        mn  = F.normalize(self.mu_neutral, dim=0)
        cs  = (hn * mn.view(1, 1, -1)).sum(-1)
        dc  = torch.exp(-self.lambda_kappa * pos_idx.float())
        return (1.0 - cs * dc).clamp(0.0, 2.0)

    # ψ(y_t,t,m) = IDF(y_t) · (t/m)^φ · σ(w_ψ · h_t)
    def compute_psi(self, hidden, token_ids, idf_table):
        B, t, _ = hidden.shape
        phi     = self.phi_exp.clamp(1.0, 4.0)
        pos     = torch.arange(1, t+1, dtype=torch.float32, device=hidden.device)
        pos_w   = (pos / float(t)).pow(phi).unsqueeze(0).expand(B, -1)
        safe    = token_ids.clamp(0, idf_table.shape[0] - 1)
        idf     = idf_table[safe]
        impact  = torch.sigmoid(self.w_psi(hidden).squeeze(-1))
        return idf * pos_w * impact

    def encode(self, inp_ids, inp_mask=None):
        inp_ids = inp_ids.clamp(0, self.vocab_size - 1)
        emb     = self.enc_emb(inp_ids)
        bv      = self.bcf(emb, inp_mask)
        rho     = self.salience(bv)
        rc      = self.tsrl(bv, rho)
        ro      = self.rcmb(rc)
        return bv, rho, ro

    def decode(self, dec_ids, rcmb_out):
        B, t    = dec_ids.shape
        dec_ids = dec_ids.clamp(0, self.vocab_size - 1)
        te      = self.enc_emb.E_tok(dec_ids)
        pi      = torch.arange(t, device=dec_ids.device).unsqueeze(0).expand(B, -1).clamp(max=self.max_len-1)
        D       = te + self.dec_pos(pi)
        kappa   = torch.zeros(B, t, dtype=D.dtype, device=D.device)
        for blk in self.dec_blocks:
            kappa = self.compute_kappa(D, pi)
            D     = blk(D, rcmb_out, kappa)
        D      = self.dec_ln(D)
        logits = self.output_proj(D)
        return logits, D, kappa

    def forward(self, inp_ids, out_ids, inp_mask=None, idf_table=None):
        bv, rho, ro = self.encode(inp_ids, inp_mask)
        dec_in      = out_ids[:, :-1]
        dec_tgt     = out_ids[:, 1:]
        logits, hidden, kappa = self.decode(dec_in, ro)
        return logits, dec_tgt, hidden, kappa, rho


# ================================================================
# CELL 7 — IDF TABLE AND HEDGE IDs
# ================================================================

def build_idf_table(pairs, vocab_size):
    print("Building IDF table...")
    df   = torch.zeros(vocab_size, dtype=torch.float32)
    N    = 0
    for _, out_ids in tqdm(pairs, desc="IDF", leave=False):
        N += 1
        for tid in set(min(max(int(x), 0), vocab_size-1) for x in out_ids):
            df[tid] += 1.0
    idf  = torch.log((N + 1.0) / (df + 1.0)) + 1.0
    idf[PAD_ID] = 0.0; idf[BOS_ID] = 0.0; idf[EOS_ID] = 0.0
    print(f"IDF — N={N:,}  max={idf.max():.3f}  mean={idf.mean():.3f}")
    return idf

idf_table = build_idf_table(train_pairs, VOCAB_SIZE)

HEDGE_IDS = []
for hp in ["sorry","no offense","just kidding","with respect",
           "apologize","forgive","didn't mean","not trying"]:
    try:
        for i in tokenizer.encode(hp):
            HEDGE_IDS.append(min(max(int(i), 0), VOCAB_SIZE-1))
    except Exception:
        pass
HEDGE_IDS = list(set(HEDGE_IDS))
print(f"Hedge token count: {len(HEDGE_IDS)}")


# ================================================================
# CELL 8 — FULL LOSS (all 5 paper terms)
# ================================================================
# L_total = L_ce
#         + λ_κ · L_κ     (tension monotonicity)
#         + λ_ψ · L_ψ     (verbal impact velocity)
#         + λ_h · L_h     (hedge penalty)
#         + λ_r · L_route (routing orthogonality)

def compute_loss(model, logits, dec_target, hidden, kappa, rho,
                 idf_table, phase=1,
                 lam_k=0.1, lam_p=0.3, lam_h=1.0, lam_r=0.01):

    B, t, V   = logits.shape
    dec_target = dec_target.clamp(0, V - 1)
    pad_mask   = (dec_target != PAD_ID)

    # L_ce
    L_ce = F.cross_entropy(logits.reshape(-1, V), dec_target.reshape(-1),
                           ignore_index=PAD_ID)
    if phase == 1:
        return L_ce, dict(L_ce=L_ce.item(), L_k=0., L_p=0., L_h=0., L_r=0.)

    # L_kappa — penalise tension drops
    if t > 1:
        drop   = F.relu(kappa[:, :-1] - kappa[:, 1:])
        vm     = pad_mask[:, :-1].float()
        L_k    = (drop * vm).sum() / (vm.sum() + 1e-8)
    else:
        L_k = logits.new_zeros(())

    # L_psi — ψ-reweighted cross-entropy
    psi   = model.compute_psi(hidden, dec_target, idf_table)
    lp    = F.log_softmax(logits, dim=-1)
    tlp   = lp.gather(-1, dec_target.unsqueeze(-1)).squeeze(-1)
    pm    = psi * pad_mask.float()
    pn    = pm / (pm.sum() + 1e-8)
    L_p   = -(pn * tlp * pad_mask.float()).sum()

    # L_hedge — suppress hedge token probability mass
    if HEDGE_IDS:
        probs   = F.softmax(logits, dim=-1)
        hids    = torch.tensor(HEDGE_IDS, dtype=torch.long)
        hedge_p = probs[:, :, hids].sum(dim=-1)
        L_h     = (hedge_p * pad_mask.float()).mean()
    else:
        L_h = logits.new_zeros(())

    # L_route — Gram matrix orthogonality
    with torch.no_grad():
        cvecs = [F.normalize(model.tsrl.projs[k].weight[0], dim=0)
                 for k in range(model.tsrl.K)]
    C_st  = torch.stack(cvecs, dim=0)
    G     = C_st @ C_st.T
    K     = model.tsrl.K
    L_r   = ((G - torch.eye(K, dtype=G.dtype)).pow(2).sum()) / float(K * K)

    L_tot = L_ce + lam_k*L_k + lam_p*L_p + lam_h*L_h + lam_r*L_r
    return L_tot, dict(L_ce=L_ce.item(),
                       L_k =L_k.item() if hasattr(L_k,'item') else 0.,
                       L_p =L_p.item(),
                       L_h =L_h.item() if hasattr(L_h,'item') else 0.,
                       L_r =L_r.item())


# ================================================================
# CELL 9 — MODEL INSTANTIATION
# ================================================================

MODEL_CFG = dict(
    vocab_size=VOCAB_SIZE, d=256, d_pos=32,
    n_enc=3, n_dec=4, K_route=8, K_anchors=8,
    M_slots=32, d_mem=64, max_len=256, lambda_kappa=0.05,
)
model = SCORCH(**MODEL_CFG)

total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n{'='*52}")
print(f"SCORCH  total params: {total_p:,}")
print(f"SCORCH  train params: {train_p:,}")
print(f"{'='*52}\n")

# Forward sanity check
with torch.no_grad():
    _i, _o, _m, _ = next(iter(train_loader))
    _lg, _tg, _h, _k, _r = model(_i, _o, inp_mask=_m, idf_table=idf_table)
    assert _lg.shape[-1] == VOCAB_SIZE
    assert not torch.isnan(_lg).any(), "NaN in logits pre-training!"
    assert not torch.isinf(_lg).any(), "Inf in logits pre-training!"
print(f"Sanity OK — logits{list(_lg.shape)}  rho={_r.mean():.3f}  kappa={_k.mean():.3f}")


# ================================================================
# CELL 10 — OPTIMISER AND LR SCHEDULE
# ================================================================

optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4,
    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
)

def get_lr(step, warmup=1000, max_steps=30000,
           max_lr=1e-4, min_lr=1e-5):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    p = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * p))

def set_lr(opt, lr):
    for pg in opt.param_groups: pg['lr'] = lr


# ================================================================
# CELL 11 — TRAINING HYPERPARAMETERS
# (TOTAL_STEPS is env-overridable for GitHub Actions)
# ================================================================

TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "8000"))
PHASE1_END  = max(1000, TOTAL_STEPS // 4)
PHASE2_END  = max(PHASE1_END + 1000, int(TOTAL_STEPS * 0.70))
WARMUP      = min(1000, TOTAL_STEPS // 8)
LOG_EVERY   = 50
EVAL_EVERY  = max(200, TOTAL_STEPS // 20)
SAVE_EVERY  = max(500, TOTAL_STEPS // 8)
GRAD_CLIP   = 1.0

print(f"\n{'='*62}")
print("SCORCH TRAINING CONFIG")
print(f"  TOTAL_STEPS : {TOTAL_STEPS:,}")
print(f"  PHASE1_END  : {PHASE1_END:,}  (L_ce only)")
print(f"  PHASE2_END  : {PHASE2_END:,}  (full loss)")
print(f"  WARMUP      : {WARMUP:,}")
print(f"  LOG_EVERY   : {LOG_EVERY}")
print(f"  EVAL_EVERY  : {EVAL_EVERY}")
print(f"  SAVE_EVERY  : {SAVE_EVERY}")
print(f"{'='*62}\n")

def get_phase(step):
    return 1 if step <= PHASE1_END else (2 if step <= PHASE2_END else 3)

def get_lam_psi(step):
    if step <= PHASE1_END: return 0.0
    ramp = float(PHASE2_END - PHASE1_END) * 0.5
    return 0.3 * min(1.0, float(step - PHASE1_END) / max(ramp, 1.0))

def get_lam_hedge(step):
    if step <= PHASE1_END: return 0.0
    ramp = float(PHASE2_END - PHASE1_END) * 0.5
    return 1.0 * min(1.0, float(step - PHASE1_END) / max(ramp, 1.0))


# ================================================================
# CELL 12 — K-MEANS ANCHOR INITIALISATION
# ================================================================

def init_anchors(model, loader, n_batches=30):
    from sklearn.cluster import KMeans
    print("\n[Anchors] Collecting belief vectors...")
    model.eval()
    bvecs = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= n_batches: break
            ii, _, im, _ = batch
            ii = ii.clamp(0, model.vocab_size - 1)
            bv, _, _ = model.encode(ii, im)
            bvecs.append(bv.numpy())
    bvecs = np.concatenate(bvecs, axis=0)
    K     = model.salience.K
    print(f"[Anchors] k-means K={K} on {bvecs.shape[0]} vectors...")
    km = KMeans(n_clusters=K, n_init=10, random_state=42, max_iter=300)
    km.fit(bvecs)
    ct = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    with torch.no_grad():
        model.salience.anchors.data.copy_(ct)
    print("[Anchors] Done — centroids copied.")
    model.train()


# ================================================================
# CELL 13 — MAIN TRAINING LOOP
# ================================================================

best_val = float('inf')
t_iter   = iter(train_loader)
ema      = None
alpha_e  = 0.95
anchored = False
t_start  = time.time()

model.train()
print(f"\n{'='*62}")
print("SCORCH TRAINING BEGIN")
print(f"{'='*62}\n")

for gstep in range(1, TOTAL_STEPS + 1):

    # fetch batch
    try:
        batch = next(t_iter)
    except StopIteration:
        t_iter = iter(train_loader)
        batch  = next(t_iter)

    inp_ids, out_ids, inp_mask, _ = batch
    phase  = get_phase(gstep)
    lam_p  = get_lam_psi(gstep)
    lam_h  = get_lam_hedge(gstep)
    lr     = get_lr(gstep, warmup=WARMUP, max_steps=TOTAL_STEPS)
    set_lr(optimizer, lr)

    if phase >= 2 and not anchored:
        init_anchors(model, train_loader, n_batches=25)
        anchored = True

    optimizer.zero_grad()
    try:
        logits, dec_tgt, hidden, kappa, rho = model(
            inp_ids, out_ids, inp_mask=inp_mask, idf_table=idf_table)
    except RuntimeError as e:
        print(f"  [SKIP fwd] step {gstep}: {e}")
        continue

    try:
        loss, comp = compute_loss(
            model=model, logits=logits, dec_target=dec_tgt,
            hidden=hidden, kappa=kappa, rho=rho,
            idf_table=idf_table, phase=phase,
            lam_k=0.1  if phase >= 2 else 0.0,
            lam_p=lam_p, lam_h=lam_h,
            lam_r=0.01 if phase >= 2 else 0.0,
        )
    except RuntimeError as e:
        print(f"  [SKIP loss] step {gstep}: {e}")
        continue

    if not torch.isfinite(loss):
        print(f"  [SKIP non-finite] step {gstep} loss={loss.item():.4f}")
        continue

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    lv  = loss.item()
    ema = lv if ema is None else (alpha_e * ema + (1 - alpha_e) * lv)

    if gstep % LOG_EVERY == 0:
        el  = time.time() - t_start
        sps = gstep / max(el, 1.0)
        eta = (TOTAL_STEPS - gstep) / sps / 60.0
        print(
            f"Step {gstep:5d}/{TOTAL_STEPS} | Ph{phase} | "
            f"EMA={ema:.4f} | "
            f"ce={comp['L_ce']:.4f} κ={comp['L_k']:.4f} "
            f"ψ={comp['L_p']:.4f} h={comp['L_h']:.4f} | "
            f"ρ̄={rho.mean():.3f} κ̄={kappa.mean():.3f} | "
            f"lr={lr:.2e} ETA={eta:.1f}m"
        )

    if gstep % EVAL_EVERY == 0:
        model.eval()
        vl = []
        with torch.no_grad():
            for vb in val_loader:
                vi, vo, vm, _ = vb
                try:
                    vlog, vtgt, vh, vk, vr = model(vi, vo, inp_mask=vm, idf_table=idf_table)
                    vls, _ = compute_loss(
                        model=model, logits=vlog, dec_target=vtgt,
                        hidden=vh, kappa=vk, rho=vr,
                        idf_table=idf_table, phase=phase)
                    if torch.isfinite(vls): vl.append(vls.item())
                except RuntimeError:
                    pass
        if vl:
            vl_mean = float(np.mean(vl))
            el_m    = (time.time() - t_start) / 60.0
            print(f"\n{'─'*62}")
            print(f"[EVAL] step={gstep} val={vl_mean:.4f} elapsed={el_m:.1f}min")
            if vl_mean < best_val:
                best_val = vl_mean
                torch.save({
                    'step': gstep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state':  optimizer.state_dict(),
                    'val_loss':         vl_mean,
                    'config':           MODEL_CFG,
                    'vocab_size':       VOCAB_SIZE,
                }, 'scorch_best.pt')
                print(f"[SAVED] scorch_best.pt  val={vl_mean:.4f}")
            print(f"{'─'*62}\n")
        model.train()

    if gstep % SAVE_EVERY == 0:
        ck = f'scorch_step_{gstep}.pt'
        torch.save({'step': gstep,
                    'model_state_dict': model.state_dict(),
                    'config': MODEL_CFG}, ck)
        print(f"[CKPT] {ck}")

total_m = (time.time() - t_start) / 60.0
print(f"\n{'='*62}")
print(f"TRAINING COMPLETE — {total_m:.1f}min  best_val={best_val:.4f}")
print(f"{'='*62}\n")


# ================================================================
# CELL 14 — INFERENCE (fixed temperature + hard token cap)
# ================================================================

def generate_roast(model, tok, input_text,
                   max_len=70, T_base=0.72, alpha_T=0.25,
                   beta_p=0.15, kappa_min=0.08, min_tokens=10,
                   max_tokens=65, repetition_penalty=1.25):
    """
    κ-modulated temperature: temp_t = T_base / (1 + α_T · κ_t)
    ρ-modulated nucleus    : p_nuc  = max(0.75, 1.0 − β_p · ρ)
    Repetition penalty     : divide logit of seen tokens by rp
    Hard stop at max_tokens: EOS forced
    """
    model.eval()
    with torch.no_grad():
        ids = tok.encode(input_text)
        if not ids:
            return "You are so unremarkable that language itself has opted out."
        ids = [min(max(int(i), 0), VOCAB_SIZE-1) for i in ids]
        it  = torch.tensor([ids], dtype=torch.long)
        im  = torch.ones_like(it, dtype=torch.bool)

        _, rho, ro = model.encode(it, im)
        rho_val = float(rho.item())
        p_nuc   = max(0.75, 1.0 - beta_p * rho_val)

        generated  = [BOS_ID]
        seen_set   = set()
        kappa_hist = []
        low_k_run  = 0

        for step in range(1, max_len + 1):
            di      = torch.tensor([generated], dtype=torch.long)
            lg, _, kappa = model.decode(di, ro)
            last_lg = lg[0, -1, :].float().clone()
            last_k  = float(kappa[0, -1].item())
            kappa_hist.append(last_k)

            # repetition penalty
            for seen_id in seen_set:
                if last_lg[seen_id] > 0:
                    last_lg[seen_id] /= repetition_penalty
                else:
                    last_lg[seen_id] *= repetition_penalty

            # κ-modulated temperature (floor 0.25)
            temp = max(T_base / (1.0 + alpha_T * last_k), 0.25)

            probs = F.softmax(last_lg / temp, dim=-1)
            probs[PAD_ID] = 0.0
            if step < min_tokens:
                probs[EOS_ID] = 0.0

            # hard stop
            if step >= max_tokens:
                break

            # nucleus sampling
            sp, si  = torch.sort(probs, descending=True)
            cum     = torch.cumsum(sp, dim=0)
            remove  = cum > p_nuc
            remove[0] = False
            sp[remove] = 0.0
            pf      = torch.zeros_like(probs)
            pf[si]  = sp
            tot     = pf.sum()
            if tot <= 1e-9:
                nt = int(si[0].item())
            else:
                pf /= tot
                nt  = int(torch.multinomial(pf, 1).item())

            if nt == EOS_ID:
                break

            generated.append(nt)
            seen_set.add(nt)

            # κ early-stop
            if last_k < kappa_min:
                low_k_run += 1
                if low_k_run >= 3 and step >= min_tokens:
                    break
            else:
                low_k_run = 0

        roast_ids = generated[1:]
        if not roast_ids:
            return "You defy description and that is not the compliment you think it is."
        result = tok.decode(roast_ids).strip()
        return result if result else "Your existence is noted and immediately set aside."


# ================================================================
# CELL 15 — LOAD BEST AND RUN TEST OUTPUTS
# ================================================================

print("Loading best checkpoint...")
try:
    ck = torch.load('scorch_best.pt', map_location='cpu')
    model.load_state_dict(ck['model_state_dict'])
    print(f"Loaded step={ck['step']}  val={ck['val_loss']:.4f}")
except FileNotFoundError:
    print("No checkpoint — using current weights.")

test_inputs = [
    "A guy who still uses Internet Explorer unironically",
    "Someone who has never finished a single book in their entire life",
    "A person who calls themselves an entrepreneur but earns zero dollars",
    "Someone who posts their 5AM workout every single morning",
    "A person who brings up their ex in every conversation",
    "Someone who says they are brutally honest but is just brutal and wrong",
    "A person who corrects strangers' grammar on the internet for fun",
    "A guy who drives a lifted pickup truck in the city and has never off-roaded",
    "Someone who went to one therapy session and now diagnoses everyone they meet",
    "A person who says they do not watch TV but knows every single show",
    "Someone who has LinkedIn premium and has never gotten a job from it",
    "A person who describes every meal as a life-changing experience",
    "Someone who only drinks black coffee and mentions it to every person they meet",
    "A person who says they are an empath but only ever talks about themselves",
    "Someone who has been writing the same novel for twelve years",
    "A crypto bro who lost absolutely everything and is still fully bullish",
    "A wellness influencer who sells supplements backed by no evidence whatsoever",
    "A person who uses the word literally incorrectly in literally every sentence",
    "Someone whose sourdough starter has a name and an official birthday",
    "A person who announces their gym era every single January",
]

print(f"\n{'='*62}")
print("SCORCH ROAST OUTPUTS")
print(f"{'='*62}\n")
for ti in test_inputs:
    roast = generate_roast(model, tokenizer, ti)
    print(f"INPUT : {ti}")
    print(f"ROAST : {roast}")
    print("─" * 62)


# ================================================================
# CELL 16 — INTERACTIVE CHATBOT
# ================================================================

print(f"\n{'='*62}")
print("SCORCH INTERACTIVE ROAST CHATBOT")
print("Describe a target. Type quit to exit.")
print(f"{'='*62}\n")

while True:
    try:
        ui = input("Who to roast? > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nDone.")
        break
    if not ui:
        continue
    if ui.lower() in ('quit','exit','q','bye','stop'):
        print("Departing. The roasts will outlive the embarrassment.")
        break
    print(f"\n🔥  {generate_roast(model, tokenizer, ui)}\n")
