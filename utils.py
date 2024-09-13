import json
import random
import string
import torch
import warnings
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load keywords
with open(Path(__file__).parent / 'data' / 'llm_20_questions' / 'keywords.py', 'r') as f:
    KEYWORDS_JSON = f.read().split('KEYWORDS_JSON = """')[1].split('"""')[0]

keywords_list = json.loads(KEYWORDS_JSON)

# Initialize global variables
device = None
model = None
tokenizer = None
model_initialized = False

# Game constants
ERROR = "ERROR"
DONE = "DONE"
INACTIVE = "INACTIVE"
ACTIVE = "ACTIVE"
TIMEOUT = "TIMEOUT"
GUESS = "guess"
ASK = "ask"
GUESSER = "guesser"
ANSWERER = "answerer"

def select_keyword():
    keyword_cat = random.choice(keywords_list)
    category = keyword_cat["category"]
    keyword_obj = random.choice(keyword_cat["words"])
    keyword = keyword_obj["keyword"]
    alts = keyword_obj["alts"]
    return category, keyword, alts

def guesser_agent(obs, debug=False):
    info_prompt = """You are playing a game of 20 questions where you ask the questions and try to figure out the keyword, which will be a real or fictional person, place, or thing. \nHere is what you know so far:\n{q_a_thread}"""
    questions_prompt = """Ask one yes or no question."""
    guess_prompt = """Guess the keyword. Only respond with the exact word/phrase. For example, if you think the keyword is [paris], don't respond with [I think the keyword is paris] or [Is the keyword Paris?]. Respond only with the word [paris]."""

    q_a_thread = ""
    for i in range(min(len(obs.questions), len(obs.answers))):
        q_a_thread += f"Q: {obs.questions[i]} A: {obs.answers[i]}\n"

    prompt = ""
    if obs.turnType == ASK:
        prompt = f"{info_prompt.format(q_a_thread=q_a_thread)}{questions_prompt}"
    elif obs.turnType == GUESS:
        prompt = f"{info_prompt.format(q_a_thread=q_a_thread)}{guess_prompt}"
    else:
        return ""
    
    if debug:
        print(f"Guesser prompt: {prompt}")
    
    response = call_llm(prompt, debug)
    
    if debug:
        print(f"Guesser response: {response}")
    
    return response

def answerer_agent(obs, debug=False):
    info_prompt = """You are a very precise answerer in a game of 20 questions. The keyword that the questioner is trying to guess is [the {category} {keyword}]. """
    answer_question_prompt = """Answer the following question with only yes, no, or if unsure maybe: {question}"""

    if obs.turnType == "answer" and obs.questions:
        prompt = f"{info_prompt.format(category=obs.category, keyword=obs.keyword)}{answer_question_prompt.format(question=obs.questions[-1])}"
        
        if debug:
            print(f"Answerer prompt: {prompt}")
        
        response = call_llm(prompt, debug)
        
        if debug:
            print(f"Answerer response: {response}")
        
        return response
    else:
        return ""

def keyword_guessed(guess: str, keyword: str, alts: list) -> bool:
    def normalize(s: str) -> str:
        t = str.maketrans("", "", string.punctuation)
        return s.lower().replace("the", "").replace(" ", "").translate(t)

    if normalize(guess) == normalize(keyword):
        return True
    for s in alts:
        if normalize(s) == normalize(guess):
            return True
    return False

def call_llm(prompt: str, debug=False) -> str:
    global model_initialized, device, model, tokenizer

    if not model_initialized:
        model_path = Path(__file__).parent / 'models' / 'flan-t5-large'
        if model_path.exists():
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = T5ForConditionalGeneration.from_pretrained(str(model_path)).to(device)
            tokenizer = T5Tokenizer.from_pretrained(str(model_path), clean_up_tokenization_spaces=True)
            model_initialized = True
        else:
            logging.info("T5-flan model not found. Downloading and saving to 'models/flan-t5-large' directory...")
            model_name = "google/flan-t5-large"
            model_dir = Path(__file__).parent / 'models' / 'flan-t5-large'
            model_dir.mkdir(parents=True, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tokenizer.save_pretrained(str(model_dir))
            model.save_pretrained(str(model_dir))
            logging.info("Model downloaded and saved successfully.")
            model_initialized = True

    if not prompt:
        raise ValueError("Empty prompt provided to call_llm")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    inputs = inputs.to(device)
    
    try:
        outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception as e:
        logging.error(f"Error in model.generate: {str(e)}")
        answer = "Error: Unable to generate response"

    if debug:
        print(f"LLM generated: {answer}")

    return answer

class Observation:
    def __init__(self, role):
        self.role = role
        self.turnType = "ask" if role == GUESSER else "answer"
        self.questions = []
        self.answers = []
        self.guesses = []
        self.keyword = None
        self.category = None
        self.step = 0

    def add_question(self, question):
        self.questions.append(question)

    def add_answer(self, answer):
        self.answers.append(answer)

    def add_guess(self, guess):
        self.guesses.append(guess)

class Agent:
    def __init__(self, role):
        self.observation = Observation(role)
        self.action = None
        self.reward = 0
        self.status = ACTIVE if role == GUESSER else INACTIVE

    def update_turn(self):
        if self.observation.role == GUESSER:
            self.observation.turnType = GUESS if self.observation.turnType == ASK else ASK