# v0idgpt — NeuronForge standalone model (Python)

Exported from NeuronForge on 2026-10-16T08:05:13.683Z. Runs the trained model without NeuronForge using a self-contained NumPy implementation of every layer (runtime v1.3.0).

## Install

```bash
pip install -r requirements.txt   # numpy
python neuronforge_model.py --verify
python neuronforge_model.py --info
python neuronforge_model.py --generate "hello there" --max-tokens 40 --temperature 0.7 --top-k 50
```

## Usage

```python
from neuronforge_model import load_model

model = load_model()  # reads bundle.json next to neuronforge_model.py

print(model.generate("hello there", max_tokens=40, temperature=0.7, top_k=50, include_prompt=True))
print(model.next_token("hello there")["top"][:5])
```

## Inputs and outputs

- **Task:** `language`
- **Input shape:** `[64]` (batch dimension excluded)
- **Output shape:** `[320]`, final activation `softmax`
- **Tokenizer:** bpe-level, 320 tokens, context 64, pad token `" "`, no end-of-sequence token
  - Text is lowercased and split into whitespace / non-whitespace runs. The t,h,th,e,i,n,a,n,r,e,a,t,o,u,e,r,o,r,i,s,o,n,e,s,a,r,e,n,l,e,y,ou,an,d,e,l,in,g,l,o,m,e,t,o,w,h,i,t,o,f,r,a,s,t,a,y,o,d,e,d,s,e,a,:,b,e,q,:,r,o,a,l,c,h,n,e,t,i,wh,at,a,re,d,o,f,or,c,o,a,d,a,s,c,e,d,ay,l,i,w,or,g,h,ra,in,t,er,w,e,i,l,le,ar,s,h,t,s,u,r,v,er,e,,,e,ar,lear,n,u,n,e,x,t,en,v,e,b,u,c,an,g,o,m,b,m,od,mod,el,p,e,s,o,a,m,h,el,i,r,k,e,n,o,ver,y,w,i,a,g,a,p,al,l,at,a,d,i,e,p,go,od,h,o,l,d,mb,er,p,l,r,i,s,s,t,rain,you,r,b,o,d,ata,e,very,en,t,f,ro,gh,t,hel,lo,lo,w,m,o,p,u,s,i,wor,k,f,o,fro,m,on,e,s,,,s,.,th,at,ti,on,y,,,a,ve,c,k,c,on,d,e,es,t,g,ra,h,e,l,a,m,a,o,p,ou,gh,ou,n,s,m,s,un,t,u,t,w,ti,me,u,l,w,el,wi,th,co,m,co,me,d,es,day,,,e,.,ear,th,es,.,for,m,i,on,ing,.,ne,t,or,n,ou,r,p,le,r,ough,re,ad,re,d,s,es,t,a,t,y,th,rough,u,se,a,me,an,s,ar,d,at,er,ay,s,b,le,bu,t,ce,.,e,k,e,t,ex,t,fo,re,g,u,h,i,ho,w,li,ke,lo,ss,n,ame,n,s,n,u,ne,ur,neur,on,nu,mber,on,g,p,er,pl,an,sm,all,t,e,t,ext,t,ur,th,an,tw,o,w,ater,wel,come,y,ear,you,.,a,se,a,use,ad,d,at,.,at,ion,at,ter,bu,il,c,t,c,ti,c,ts,ch,in,d,.,day,.,e,re,earth,.,en,.,er,.,es,,,ex,am,f,i,f,ri,g,re,gra,di,h,ave,i,ght,i,le,in,to,is,t,it,.,it,t,k,es,li,ght,low,er,m,orn,mo,re,net,work,no,t,o,k,o,l,oun,d,p,as,p,ro,re,c,s,or,st,ar,t,,,ten,sor,th,is,time,.,u,re,u,s,v,el,w,er,w,r,we,ek,wh,en,wh,ere,wor,ld,a,in,a,ir,ag,ain,al,k,al,w,alw,ays,an,g,ans,wer,ar,g,as,ure,at,s,b,a,b,lo,b,r,b,y,be,at.,be,c,be,fore,bec,ause,bo,ok,c,lo,c,tion,ch,e,chin,e,co,l,col,or,com,pu,compu,ter,d,s,day,?,des,c,di,d learned BPE merge rules are re-applied, so subword tokenisation matches training exactly; unseen characters become the `<unk>` token (id 0).
  - Prompts longer than 64 tokens are truncated to the most recent 64; shorter ones are left-padded.

## Verification and numerical behaviour

`--verify` replays test vectors computed by NeuronForge at export time. NumPy's matrix multiply accumulates in a different order from the browser engine, so forward outputs are compared with a tolerance of 1e-4 absolute + 1e-3 relative rather than bit-for-bit. The sampler and random generator are ports of NeuronForge's, so seeded generation usually reproduces the same tokens; a rare divergence (a near-tie flipped by float rounding) is reported as a warning.

**This Python runtime was generated, not executed, by NeuronForge** — the browser cannot run Python. Run `--verify` before relying on it.

## What this export does not include

- Newline ("\n") is treated as end-of-sequence by default (stopAtEos). Pass stop: [] to disable.
- Inference only: the export cannot train, fine-tune, or compute gradients, and contains no optimiser state. Use “Download checkpoint” to resume training inside NeuronForge.
- BatchNorm uses its running statistics and Dropout is disabled, matching NeuronForge's evaluation behaviour.
- Exporting preserves the learned behaviour; it does not make the model more capable.
- The training dataset is included as dataset.json (optional export).
- ONNX export is not implemented.
