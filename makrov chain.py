import random
from collections import defaultdict

class CharacterMarkovChainTextGenerator:
    def __init__(self, n=3):
        """
        n: The number of characters to use as the state (i.e., the "order" of the chain).
        """
        self.n = n
        self.model = defaultdict(list)

    def train(self, text):
        """
        Trains the Markov model using the provided text.
        The model stores sequences of n characters and the next character that follows them.
        """
        for i in range(len(text) - self.n):
            state = text[i:i + self.n]
            next_char = text[i + self.n]
            self.model[state].append(next_char)

    def generate_text(self, seed, length=500):
        """
        Generates text starting with the given seed.
        length: The total number of characters to generate.
        """
        current_state = seed[:self.n]
        result = [current_state]

        for _ in range(length):
            next_chars = self.model.get(current_state, None)
            if not next_chars:
                break  # If no next characters are found, stop generation
            next_char = random.choice(next_chars)
            result.append(next_char)

            # Update the current state to the last n characters generated
            current_state = ''.join(result)[-self.n:]

        return ''.join(result)

if __name__ == "__main__":
    # Sample text for training (can be a large block of text, book, or article)
    sample_text = """
    In a kingdom far, far away, a young princess dreamt of exploring the vast world beyond
    her castle. Every night, she would look up at the stars and wonder what lay beyond
    the mountains and seas.
    """

    # Create an instance of the character-level Markov generator
    generator = CharacterMarkovChainTextGenerator(n=4)
    generator.train(sample_text)

    # Generate text starting with a seed (e.g., first few characters of the training text)
    seed_text = "In a"
    generated_text = generator.generate_text(seed=seed_text, length=500)

    print(generated_text)
