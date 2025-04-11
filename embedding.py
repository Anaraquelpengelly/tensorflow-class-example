import tensorflow_hub as hub
from tensorflow import string

class myEmbeddings:
    def __init__(self, model):
        self.model = "https://tfhub.dev/google/nnlm-en-dim50/2"
        pass

    @property
    def embedLayer(self):
        embed_l =  hub.keras_layer(self.model, input_shape=[], dtype=string, trainable=True)
        return embed_l