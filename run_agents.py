"""
This module implements the main game logic for a 20 Questions-style game using AI agents.

The game involves two agents: a Guesser and an Answerer. The Answerer knows a secret keyword,
and the Guesser tries to guess it by asking up to 20 yes/no questions. The module handles
the turn-based interaction between these agents and determines the game's outcome.
"""

from utils import (
    GUESSER, ANSWERER, ACTIVE, INACTIVE, DONE, ASK, GUESS,
    select_keyword, guesser_agent, answerer_agent, Agent, keyword_guessed
)
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_game(debug=False):
    """
    Execute a single game of 20 Questions between AI agents.

    This function manages the game flow, including keyword selection, agent initialization,
    turn-taking, and game termination conditions.

    Args:
        debug (bool): If True, enables debug mode for more verbose output.

    Returns:
        None
    """
    # Select a keyword
    category, keyword, alts = select_keyword()
    logging.info(f"Selected keyword: {keyword} (Category: {category})")
    
    # Initialize agents
    guesser = Agent(GUESSER)
    answerer = Agent(ANSWERER)
    answerer.observation.keyword = keyword
    answerer.observation.category = category

    # Main game loop
    while True:
        # Guesser's turn
        if guesser.status == ACTIVE:
            if guesser.observation.turnType == ASK:
                question = guesser_agent(guesser.observation, debug)
                logging.info(f"Guesser asks: {question}")
                guesser.observation.add_question(question)
                answerer.observation.add_question(question)
                guesser.update_turn()
            elif guesser.observation.turnType == GUESS:
                guess = guesser_agent(guesser.observation, debug)
                logging.info(f"Guesser guesses: {guess}")
                guesser.observation.add_guess(guess)
                if keyword_guessed(guess, keyword, alts):
                    logging.info(f"Correct! The keyword was {keyword}.")
                    guesser.reward = 20 - len(guesser.observation.questions)
                    break
                guesser.update_turn()
            guesser.status = INACTIVE
            answerer.status = ACTIVE

        # Answerer's turn
        elif answerer.status == ACTIVE:
            answer = answerer_agent(answerer.observation, debug)
            logging.info(f"Answerer responds: {answer}")
            guesser.observation.add_answer(answer)
            answerer.observation.add_answer(answer)
            answerer.status = INACTIVE
            guesser.status = ACTIVE

        # Check if maximum questions reached
        if len(guesser.observation.questions) >= 20:
            logging.info(f"Game over. The keyword was {keyword}.")
            break

    logging.info(f"Guesser's score: {guesser.reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 20 Questions game")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    run_game(debug=args.debug)