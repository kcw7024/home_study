import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Current device:', device)

model = AutoModelForCausalLM.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
  pad_token_id=tokenizer.eos_token_id,
  torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device=device, non_blocking=True)
_ = model.eval()

# 입력 문장(prompt)을 받아 모델에서 생성된 결과를 보여주는 함수를 만듭니다.
def gpt(prompt, max_length: int = 256):
    with torch.no_grad():
        # 입력문장을 토크나이저를 사용하여 토큰화
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device=device, non_blocking=True)
        # 토큰화된 문장을 입력으로 토큰형태의 새로운 문장 생성
        gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=max_length)
        # 생성된 문장을 다시 문자열 형태로 디코딩
        generated = tokenizer.batch_decode(gen_tokens)[0]
    return generated

prompt = """인간처럼 생각하고, 행동하는 '지능'을 통해 인류가 이제까지 풀지 못했던"""
gpt(prompt)
