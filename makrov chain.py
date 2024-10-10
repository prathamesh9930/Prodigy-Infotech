import random
import string

class MarkovChainTextGenerator:
    def __init__(self, n=1):
        self.n = n
        self.model = {}

    def preprocess_text(self, text):
        # Remove punctuation and lower the case
        return text.translate(str.maketrans('', '', string.punctuation)).lower().split()

    def train(self, text):
        tokens = self.preprocess_text(text)
        for i in range(len(tokens) - self.n):
            state = tuple(tokens[i:i + self.n])
            next_token = tokens[i + self.n]
            if state not in self.model:
                self.model[state] = []
            self.model[state].append(next_token)

    def generate_text(self, seed, length=50):
        current_state = tuple(self.preprocess_text(seed)[-self.n:])
        result = list(current_state)
        for _ in range(length):
            next_tokens = self.model.get(current_state, None)
            if not next_tokens:
                break
            next_token = random.choice(next_tokens)
            result.append(next_token)
            current_state = tuple(result[-self.n:])
        return ' '.join(result)

if __name__ == "__main__":
    # Sample text for training
    sample_text = """
    This is a sample text for training the Markov chain text generator. The generator
    will attempt to create new text that is similar to this sample text.
    """
    
    # Creating an instance of the generator with order 1 (unigram)
    generator = MarkovChainTextGenerator(n=1)
    generator.train(sample_text)

    # Generating text with a seed
    seed_text = "This is"
    generated_text = generator.generate_text(seed_text, length=50)
    print(generated_text)
