
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from language_modell import KenlmModel
import kenlm
import numpy as np

def path_to_mfcc(path):

    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)    

    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    
    audio_len = tf.shape(x)[0]
    
    pad_len = 2000
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    
    return x
class VectorizeChar:
    def __init__(self, unique_chars, max_len=50):
        self.vocab = (
            [
                "-",
                "#", 
                "<", # use as start token
                ">"  # use as end token
            ]
            + unique_chars
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        '''Make vectorizer as callable object on text
        Args:
            text: str, text sequence
        Returns:
            text sequence represent as number
        '''
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocab(self):
        '''Return all the available vocabulary'''
        return self.vocab
class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)
class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

class TransformerEncoder(layers.Layer):
    '''Encode the speech information'''
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerDecoder(layers.Layer):
    '''Decode the speech information'''
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm


class Transformer(keras.Model):
    '''Transform model for speech recognition'''
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=10,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    @tf.function
    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch["source"]
        print(source)
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]

        with tf.GradientTape() as tape:
            # take prediciton from the model
            preds = self([source, dec_input])

            # create one-hot and mask to computer loss
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))

            # calculate loss on training data
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        # take the all trainable variable the modle
        trainable_vars = self.trainable_variables
        # calculate gradient accordint to loss and trainable variable
        gradients = tape.gradient(loss, trainable_vars)
        
        # update trainable variable accordint to gradient calcualted
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # update loss matric
        self.loss_metric.update_state(loss)
        
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        '''Test step perform after each batch finished'''
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]

        # take predictioin from the model
        preds = self([source, dec_input])

        # create one-hot and mask to computer loss
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))

        # computer loss on test sample
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        # update loss metric
        self.loss_metric.update_state(loss)

        return {"loss": self.loss_metric.result()}
    
    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)

        return dec_input
def predict(source, target_start_token_idx, target_maxlen,nmodel):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        enc = nmodel.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        logits = None
        for i in range(target_maxlen-1):
            dec_out = nmodel.decode(enc, dec_input)
            logits = nmodel.classifier(dec_out)
            
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input,logits
def decode(chars,arr):
    snt = ""
    for ch in arr:
        snt += chars[ch]
        if chars[ch] == ">":
            break
    return snt
def spch_reg(path, model):
    target_audio = path_to_mfcc(path)
    target_audio = tf.expand_dims(target_audio, 0)
    logits = None
    # perform prediction on given target audio file
    preds,logits = predict(target_audio, 2, 40,model)

    preds = preds.numpy()[0]


    snt = decode(vectorizer.get_vocab(),preds)
    return snt,logits



with open('./bn_asr/uchars.pickle', 'rb') as f:
    # Load the Python object from the file
    loaded_obj = pickle.load(f)

vectorizer = VectorizeChar(loaded_obj,40)
# class CTCBeamDecoder:

#     def __init__(self, beam_size=100, blank_id=vectorizer.get_vocab().index('_'), kenlm_path=None):
#         print("loading beam search with lm...")
#         self.decoder = ctcdecode.CTCBeamDecoder(
#             vectorizer.get_vocab(), alpha=0.522729216841, beta=0.96506699808,
#             beam_width=beam_size, blank_id=vectorizer.get_vocab().index('_'),
#             model_path=kenlm_path)
#         print("finished loading beam search")

#     def __call__(self, output):
#         beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
#         return self.convert_to_string(beam_result[0][0], vectorizer.get_vocab(), out_seq_len[0][0])

#     def convert_to_string(self, tokens, vocab, seq_len):
#         return decode(tokens[0:seq_len])

# def beam_search_lm(sent, k=3, lm_path='./model.arpa', alpha=0.8, beta=0.3):
#     # Load the language model
#     lm = kenlm.LanguageModel(lm_path)

#     # Tokenize the input sentence
#     words = sent.strip().split()

#     # Set the initial hypothesis to the start symbol and its log probability
#     hypotheses = [([], 0)]

#     # Perform beam search
#     for i in range(len(words)):
#         # Get the logits for the next word
#         next_word = words[i]
#         next_word_logprobs = np.array([lm.score(hypo[0] + [next_word], eos=False) for hypo in hypotheses])

#         # Compute the beam score for each hypothesis
#         beam_scores = np.array([hypo[1] + next_word_logprobs[j] * alpha + (i+1) * beta for j, hypo in enumerate(hypotheses)])

#         # Select the top k hypotheses with the highest beam scores
#         topk_indices = np.argsort(beam_scores)[::-1][:k]
#         new_hypotheses = []
#         for index in topk_indices:
#             hypothesis, score = hypotheses[index]
#             new_hypothesis = (hypothesis + [next_word], score + next_word_logprobs[index] * alpha + (i+1) * beta)
#             new_hypotheses.append(new_hypothesis)
#         hypotheses = new_hypotheses

#     # Select the hypothesis with the highest score
#     best_hypothesis = max(hypotheses, key=lambda hypo: hypo[1])[0]
#     return ' '.join(best_hypothesis)




def beam_search_lm(lm_model, context, beam_size, end_token, max_len):
    """
    Performs beam search using a language model.
    :param lm_model: A KenLM language model object.
    :param context: A list of integers representing the context for the beam search.
    :param beam_size: The beam size to use for decoding.
    :param end_token: The index of the end token in the vocabulary.
    :param max_len: The maximum length of the output sequence.
    :return: A list of tuples, where each tuple is (sequence, score).
    """
    sequences = [([], 0)]  # (sequence, score)
    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            for v in range(lm_model.order + 1):
                candidate = (seq + [v], score - lm_model.get_perplexity(seq + [v], eos=False))
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=False)
        sequences = []
        for seq, score in ordered:
            if seq[-1] == end_token or len(seq) == max_len:
                sequences.append((seq, score))
            if len(sequences) == beam_size:
                break
    return sequences


model = Transformer(
    num_hid=200,
    num_head=2,
    num_feed_forward=400,
    target_maxlen=40,
    num_layers_enc=4,
    num_layers_dec=1,
    num_classes=len(vectorizer.get_vocab()), #108 -> number of vocab in vectorizer
)
latest = tf.train.latest_checkpoint("./bn_asr/")

model.load_weights(latest)


path = "./audio/0a5a1687aa.wav"

lmmodel = KenlmModel.from_pretrained("./language model", "bn")
snt,logits = spch_reg("./recording.wav",model)
snt = snt[1:-1] 
print(snt)

with open("./reslt.txt","w",encoding="UTF-8") as fl:
    fl.write(snt)