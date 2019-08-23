"""Dictionary utilities."""


class Dictionary:
    """Simple dictionary class to store text-to-index and index-to-text."""

    def __init__(self, iterable=None, pad=False, unk=False, strict=True):
        """Initialize Dictionary.

        Args:
          iterable: An iterable of tokens to build dictionary from. Should
            unk and pad if desired later on.
          pad: If True, add a PADDING token.
          unk: If True, add an UNKOWN token.
          strict: If True, throw an error when trying to access an unknown token
            when unk is False. Otherwise, return -1 without error.
        """
        self.t2i = {}
        self.i2t = []
        self.strict = strict
        if pad and iterable is None:
            self.add('<pad>')
        if unk and iterable is None:
            self.add('<unk>')
        if iterable is not None:
            for t in iterable:
                self.add(t)

    def __len__(self):
        return len(self.t2i)

    def __iter__(self):
        return iter(self.i2t)

    def __contains__(self, key):
        return key in self.t2i

    def __getitem__(self, key):
        if type(key) == int:
            return self.i2t[key]
        if type(key) == str:
            if key not in self.t2i:
                if key.upper() in self.t2i:
                    return self.t2i[key.upper()]
                if self.strict:
                    assert '<unk>' in self.t2i
                return self.t2i.get('<unk>', -1)
            return self.t2i[key]

    def __eq__(self, other):
        if self.t2i != other.t2i:
            return False
        if self.i2t != other.i2t:
            return False
        return True

    def add(self, token):
        """Add token to dictionary, if not already indexed."""
        if token not in self.t2i:
            self.t2i[token] = len(self.t2i)
            self.i2t.append(token)

    def update(self, tokens):
        """Update dictionary while iterating over tokens."""
        for t in tokens:
            self.add(t)

    def items(self):
        """Iterate over all tokens."""
        return self.t2i.items()

    def save(self, filename):
        with open(filename, 'w') as f:
            for t in self.i2t:
                f.write(t + '\n')

    @classmethod
    def load(cls, filename):
        with open(filename) as f:
            tokens = f.read().splitlines()
        return cls(tokens)
