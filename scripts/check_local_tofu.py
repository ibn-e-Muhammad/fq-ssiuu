"""Check the notebook's real loss/gradient functions without loading 1B weights."""
import ast
import json
from pathlib import Path
import importlib.metadata as metadata

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / 'Mother_Local_v2.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']), filename=f'cell_{i}')
print('All notebook cells compile.')
print({p: metadata.version(p) for p in ['torch', 'transformers', 'peft', 'datasets', 'bitsandbytes']})
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU free GiB:', round(torch.cuda.mem_get_info()[0] / 2**30, 2))

namespace = {'torch': torch, 'F': F}
tree = ast.parse(''.join(notebook['cells'][8]['source']))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'answer_logps')
exec(compile(ast.Module(body=[function], type_ignores=[]), '<answer_logps>', 'exec'), namespace)
model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj','v_proj'],
    task_type='CAUSAL_LM', lora_dropout=0.0))
batch = {'input_ids': torch.tensor([[1,2,3,4,0]]),
         'attention_mask': torch.tensor([[1,1,1,1,0]]),
         'labels': torch.tensor([[-100,-100,3,4,-100]])}
lp, count = namespace['answer_logps'](model, batch)
assert count.item() == 2
expected = model(**batch).loss
torch.testing.assert_close(-lp.sum()/count.sum(), expected)
(-lp/count).mean().backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
model.zero_grad()
lp, _ = namespace['answer_logps'](model, batch)
ref_lp = lp.detach().clone()
beta = 0.1
npo = -(2/beta)*F.logsigmoid(-beta*(lp-ref_lp)).mean()
grad, = torch.autograd.grad(npo, lp, retain_graph=True)
assert grad.item() > 0, 'Minimizing NPO must reduce forget answer log probability'
npo.backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
tokenizer = AutoTokenizer.from_pretrained('C:/Model/Llama-3.2-1B-Instruct', local_files_only=True)
assert tokenizer.apply_chat_template([{'role':'user','content':'Test'}], tokenize=False, add_generation_prompt=True)
print('PASS: answer masking, CE equivalence, SFT gradients, NPO direction and gradients, local tokenizer.')
