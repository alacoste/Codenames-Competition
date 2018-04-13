import random


class SpyBot(object):
    """This abstract class represents a bot that acts as a spy master for Codenames and can either give clues to the guesser
     or be updated for game state or get a clue."""

    def __init__(self, vocab, game_board, p_id):
        """initialization for the bot
        Args:
            vocab (List<string>): list of English words that can be used to generate clues.
            game_board (Map<string, Enum:WordType>): a dictionary representing a the game board, mapping the words on
                the board with an enum representing whether the word belongs to player 1, player 2, is neutral or a spy word.
            p_id: (int): integer (0, 1) representing the the id of the bot, player 1 or 2.
        """
        self.vocab = vocab
        pass

    def update(self, is_my_turn, clue_word, clue_num_guesses, guesses):
        """Updates the bot with the changed game state after a round.
        Args:
            is_my_turn (bool): True if it is the bot's turn to play or if it is the opponent's turn.
            clue_word (str): String representing the clue word in the round.
            clue_num_guesses (int): int representing the number of words that matched the given clue word.
            guesses: (List<string>): list of words that were guessed in the round.
        """
        pass

    def getClue(self, invalid_words):
        """Updates the bot with the changed game state after a round.
        Args:
            invalid_words (Set<string>): set of words that are not allowed to be clued. Eg. words that share a root with
                any of the words on the board.
        Returns:
            a tuple (clue, num_guesses) of a string representing the clue word which must come from
                self.vocab and must not be in the set of invalid words, and an int representing the number of words
                on the board that are supposed to that match the given clue.
        """
        pass


# example of how to write a bot
class RandomBot(SpyBot):

    def __init__(self, vocab, game_board, p_id):
        self.vocab = set(vocab)

    def getClue(self, invalid_words):
        return random.choice(list(self.vocab.difference(invalid_words))), 2