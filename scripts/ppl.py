import json
from dataclasses import dataclass


@dataclass
class Perplexity:

    ppl: float

    def __init__(self, value):
        self.ppl = value

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
