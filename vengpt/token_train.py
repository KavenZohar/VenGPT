from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from vengpt.dataset import PretrainDataset

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    initial_alphabet=ByteLevel.alphabet(),
    special_tokens=[
        "<[BOS]>",
        "<[EOS]>",
        "<[EOT]>",
        "<[{SYSTEM}]>",
        "<[{USER}]>",
        "<[{ASSISTANT}]>"
    ]
)

pd = PretrainDataset()

def text_iterator():
    for index, sample in enumerate(pd()):
        if index >= 10000:
            break

        text = sample.get("text", "")

        if text:
            yield text

tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

tokenizer.save("tokenizer.json")
