import torch
from transformers import GPT2LMHeadModel


from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
bos_token='</s>', eos_token='</s>', unk_token='<unk>',
pad_token='<pad>', mask_token='<mask>') 
tokenizer.tokenize("안녕하세요. 한국어 GPT-2 입니다.😤:)l^o")
['▁안녕', '하', '세', '요.', '▁한국어', '▁G', 'P', 'T', '-2', '▁입', '니다.', '😤', ':)', 'l^o']


model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
text = '눈위에서 노는 강아지 두마리'
input_ids = tokenizer.encode(text)
gen_ids = model.generate(torch.tensor([input_ids]),
                        max_length=50,
                        repetition_penalty=2.0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,

                        bos_token_id=tokenizer.bos_token_id,
                        use_cache=True)
generated = tokenizer.decode(gen_ids[0,:].tolist())

print(generated)