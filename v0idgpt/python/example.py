from neuronforge_model import load_model

model = load_model()  # reads bundle.json next to neuronforge_model.py

print(model.generate("hello there", max_tokens=40, temperature=0.7, top_k=50, include_prompt=True))
print(model.next_token("hello there")["top"][:5])
