#!wget --no-verbose -c https://xkcd.com/2601/radio.mp3
#!wget --no-verbose -c https://raw.githubusercontent.com/theinternetftw/xkcd2601/main/xkcd.lgo
#!pip3 install transformers[speech,sentencepiece] datasets librosa soundfile >/dev/null

print('importing libraries ...')
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import librosa as lb
import numpy as np
import inspect

import os
import sentencepiece as spm
print('imported')

class CustomTokenizer:
  def __init__(self, datafilename, vocab_size):
    self.fn = datafilename
    self.vocab_size = vocab_size
  def load(self):
    modelpfx = f'{self.fn}.{self.vocab_size}.model'
    modelfn = f'{modelpfx}.model'
    if not os.path.exists(modelfn):
      def data(chunksize):
        with open(self.fn, 'rt') as datafile:
          while True:
            chunk = datafile.read(chunksize)
            if len(chunk) < chunksize:
              break
            yield chunk
      spm.SentencePieceTrainer.train(sentence_iterator=data(1024), model_prefix=modelpfx, vocab_size=self.vocab_size, byte_fallback=True)
    self.model = spm.SentencePieceProcessor(model_file=modelfn)
  def tokenize(self, inputs):
    return self.model.encode(inputs)
  def detokenize(self, batch_ids):
    ids = [torch.where(ids < self.vocab_size, ids, torch.tensor(0)).tolist() for ids in batch_ids]
    return self.model.decode(ids)

class Data:
  def __init__(self, src = 'radio.mp3', chunksize = 80 * 6000, sr = 16_000, dtype = np.float32):
    self.src = src
    self.chunksize = chunksize
    self.sr = sr
    self.length = lb.get_duration(filename = self.src)
    self.dtype = dtype
  def chunk_duration(self, chunksize = None):
    if chunksize is None:
      chunksize = self.chunksize
    return chunksize / self.sr
  def read_one(self, offset, chunksize = None):
    duration = self.chunk_duration(chunksize)
    print(f'reading {duration}s at {offset}s ...')
    data, sr = lb.load(self.src, sr = self.sr, offset = offset, duration = duration, dtype = self.dtype)
    print(f'read {data.shape} samples at {sr}Hz')
    return data
  def read_random(self, ct=1):
    return np.stack([self.read_one(np.random.random() * (self.length - self.duration)) for idx in range(ct)])
  def read_chunks(self, ct=1, offset=0):
    chunksize = self.chunksize
    data = self.read_one(offset, chunksize * ct)
    return data.reshape((ct, chunksize))

class S2T:
  def __init__(self, model = "facebook/s2t-small-librispeech-asr", sr = 16_000, detokenizer = None):
    self.sr = sr
    self.model = Speech2TextForConditionalGeneration.from_pretrained(model)
    self.processor = Speech2TextProcessor.from_pretrained(model)
    self.detokenizer = detokenizer
  @property
  def vocab_size(self):
    return self.model.config.vocab_size
  def tokenize(self, inputs):
    print('tokenizing ...')
    input_ids = self.processor(inputs, sampling_rate=self.sr, return_tensors='pt')
    return input_ids['input_features'], input_ids['attention_mask']
  def forward(self, input_features, decoder_input_ids, attention_mask):
    # returns next logits given passed preceding, of shape [batch, sequence, vocab]
    return self.model(input_features, decoder_input_ids=decoder_input_ids, return_dict=True, attention_mask=attention_mask).logits
  def __enter__(self):
    # quick training loop?
    optimizer = torch.optim.SGD(lr=0.0001)
  def generate(self, input_features, attention_mask):
    print('passing data thru model ...')
    input_ids = torch.full((input_features.shape[0], 1), self.model.config.decoder_start_token_id)
    finished_sequences = []
    while len(input_ids):
      next_token_logits = self.forward(input_features, input_ids, attention_mask)[:, -1, :]
      next_tokens = torch.argmax(next_token_logits, dim=-1)
      input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

      just_finished_sequences = (next_tokens == self.model.config.eos_token_id)
      for sequence in input_ids[just_finished_sequences]:
        finished_sequences.append(sequence)
      input_ids = input_ids[just_finished_sequences == False]
    return finished_sequences
  def detokenize(self, generated_ids):
    print('detokenizing output ...')
    if self.detokenizer is None:
      return self.processor.batch_decode(generated_ids)
    else:
      return self.detokenizer.detokenize(generated_ids)

if __name__ == '__main__':
    import sys

    print('constructing structures...')
    s2t = S2T()
    #s2t_vanilla = S2T()
    #detokenizer = CustomTokenizer('xkcd.lgo', vocab_size=1200)#s2t.vocab_size)
    #s2t_retokenized = S2T(detokenizer = detokenizer)
    
    #detokenizer.load()
    feature_ids, attention_mask = s2t_vanilla.tokenize(data.read_chunks(1)[0])
    generated_ids = s2t_vanilla.generate(feature_ids, attention_mask)
    outputs = s2t_vanilla.detokenize(generated_ids)
    print(outputs)
    #print('empty output:', s2t_retokenized.detokenize(torch.zeros((1,4), dtype=torch.long)))
    #print('output as if by other tokenizer:', s2t_retokenized.detokenize(generated_ids))
    for fn in sys.argv[1:]:
        data = Data(src = fn)
        processed = []
        for larger_chunk in range(0, data.length - data.chunk_duration(), data.chunksize * 1024):
        for offset in range(0, data.length - data.chunk_duration(), data.length / 3):
            feature_ids, attention_mask = s2t_vanilla.tokenize(data.read_chunks(1)[0])
        generated_ids = s2t_vanilla.generate(feature_ids, attention_mask)
        outputs = s2t_vanilla.detokenize(generated_ids)
        print(outputs)

        # stitching doesn't seem that nice
        # but it is notable that we have basically a bunch of set feature ids
        # it then processes the feature ids in a set chunk size
        # so basically we would read from only the middle, and stitch that part together
        # we have a stream of feature_ids.
        # we produce more generated_ids than we use
        # and we stitch the generated_ids.
        ### missing: where to stitch them.
        # [ first block of feature ids]
        #       [second block of feature ids]
        #           [interesting ]
        # thinking of processing the area described interesting
        # after reading 2 overlapping blocks, there are overlapping
        # generated ids output. we want the generated ids that are
        # the same.
        # another approach would be to add the logits, if knew where
        # they lined up.
        # produce generated ids, slide window, produce more generated ids
        # how find identical range?
        ### mm i guess you'd put it into a slid matrix and find the row with the most equality
        # could do that with logits too. find the biggest product e.g.

        # ok so logits from one, logits from another
        ### we did not actually figure streaming out. misleadingness.
        ### it turns out the model is more complex than imagined.
        ### inside model, it may turn out that it is not hard.
        ### but would need to understand inside of model to do not-hard way.
        ## we might be able to infer some of that if we assume it's a transformer encoder feeding partway into a transformer decoder that takes the input ids, trained on example sequences that do not chain.
        ## although it may have been trained on chaining examples.

        #### we could do a quick + simple streaming but the performance
        #### of the model would reduce. it would make more sense to
        #### do the cross-correlationish-thing.
