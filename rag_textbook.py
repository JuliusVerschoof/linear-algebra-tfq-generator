import os
import re
import pickle
import numpy as np
import faiss
import openai
import toml
import pdfplumber

# ----------------- Load OpenAI Key -----------------
secrets = toml.load('.streamlit/secrets.toml')
openai.api_key = secrets.get('OPENAI_API_KEY')
if not openai.api_key:
    raise RuntimeError('OPENAI_API_KEY not found in .streamlit/secrets.toml')

# ----------------- Extract Textbook Sections -----------------
# Read entire PDF into text
pages = []
with pdfplumber.open('linear_algebra_textbook.pdf') as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
full_text = "\n\n".join(pages)

# Split into section blocks by lines that begin with a section number N.M
raw_sections = re.split(r'(?m)^(?=\d+\.\d)', full_text)
# We want sections 1.1–1.5, 1.7–1.9, and 2.1 and 2.2,2.3 4.1,3.2,4.1-4.4
desired = {
    
    *{f'1.{i}' for i in range(1, 6)},
    *{f'1.{i}' for i in range(7, 10)},
    
    *{f'2.{i}' for i in (1, 2, 3)},
 
    *{f'3.{i}' for i in (1, 2)},
 
    *{f'4.{i}' for i in range(1, 5)},
}
sections = []  # list of (section_id, text)
for sec in raw_sections:
    lines = sec.strip().splitlines()
    if not lines:
        continue
    sec_id = lines[0].split()[0]
    if sec_id in desired:
        sections.append((sec_id, sec))

# ----------------- Chunk into Paragraphs -----------------
chunks = []  # list of (chunk_id, text)
for sec_id, sec_text in sections:
    paras = [p.strip() for p in sec_text.split('\n\n') if p.strip()]
    for idx, para in enumerate(paras):
        chunk_id = f"{sec_id}#{idx}"
        chunks.append((chunk_id, para))

# ----------------- Compute Embeddings -----------------
texts = [t for (_id, t) in chunks]
embeddings = []
batch_size = 50
for i in range(0, len(texts), batch_size):
    batch = texts[i : i + batch_size]
    resp = openai.Embedding.create(
        model='text-embedding-ada-002', input=batch
    )
    embeddings.extend([d.embedding for d in resp.data])

emb_array = np.array(embeddings, dtype='float32')
faiss.normalize_L2(emb_array)

# ----------------- Build FAISS Index -----------------
dim = emb_array.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb_array)

# ----------------- Persist Index & Metadata -----------------
index_file = 'book_index.faiss'
meta_file = 'book_chunks.pkl'
faiss.write_index(index, index_file)
with open(meta_file, 'wb') as f:
    pickle.dump(chunks, f)

print(f"Built {index_file} and {meta_file}, containing {len(chunks)} chunks.")