import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.utils.generic_utils import to_list

class CompressedEmbedding(Layer):
    """Embedding layer which uses compressed embeddings."""

    def __init__(self, codebook, codes, input_length=None, **kwargs):
        """Initializes a new compressed embedding layer.

        - `codebook` is a matrix of codebooks which map indices to basis vectors
        - `codes` is a matrix which maps word indices to sequences of integers,
            representing indexes into each codebook.
        """

        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)

        assert isinstance(codebook, np.ndarray)
        assert isinstance(codes, np.ndarray)

        self.codebook_np = codebook
        self.codes_np = codes

        self.input_length = input_length
        self.output_dim = codebook.shape[-1]

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.codebook = K.constant(self.codebook_np, dtype='float32', name='codebook')
        self.codes = K.constant(self.codes_np, dtype='int32', name='word_codes')

        super().build(input_shape)

    def call(self, x):
        if K.dtype(x) != 'int32':
            x = K.cast(x, 'int32')

        # Get the indices into the codebooks
        codes = K.gather(self.codes, x)

        # Gather the required basis vectors for these words
        vectors = K.gather(self.codebook, codes)

        # Sum the basis vectors to obtain the embedding vectors
        embeddings = K.sum(vectors, axis=-2)

        return embeddings

    def compute_output_shape(self, input_shape):
        """Computes the output shape of this embedding layer.

        Code taken from the original Keras `Embedding` layer.
        """

        if self.input_length is None:
            return input_shape + (self.output_dim,)

        # input_length can be tuple if input is 3D or higher
        in_lens = to_list(self.input_length, allow_tuple=True)
        if len(in_lens) != len(input_shape) - 1:
            raise ValueError(
                '"input_length" is %s, but received input has shape %s' %
                (str(self.input_length), str(input_shape)))

        for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
            if s1 is not None and s2 is not None and s1 != s2:
                raise ValueError(
                    '"input_length" is %s, but received input has shape %s' %
                    (str(self.input_length), str(input_shape)))

            if s1 is None:
                in_lens[i] = s2

        return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)
