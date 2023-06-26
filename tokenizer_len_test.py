## len adjust for bang
from transformers import DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('neuralspace-reverie/indic-transformers-bn-distilbert')

captions = ['একটি দমকা লাল জ্যাকেটের একটি মহিলা একটি আইস স্কেটিং রিংতে ছবির জন্য পোজ দিচ্ছেন। একটি দমকা লাল জ্যাকেটের একটি মহিলা একটি আইস স্কেটিং রিংতে ছবির জন্য পোজ দিচ্ছেন।']
encoded_captions = tokenizer(
            captions, padding=True, truncation=True, max_length=5
)
print(encoded_captions)

o = tokenizer.tokenize(captions[0])
print(o)