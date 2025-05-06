# build_rag_index.py
import pickle
import numpy as np
import faiss
import openai
import toml 

# 
secrets = toml.load(".streamlit/secrets.toml")
openai.api_key = secrets["OPENAI_API_KEY"] 

if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .streamlit/secrets.toml")

# Chunk corpus
PROMPT_SECTIONS = {
    "Systems & Row Reduction": (
        "System of Linear Equations & Row Reduction:\n"
        "A system of linear equations has: no solution, or exactly one solution, or infinitely many solutions. "
        "A system is consistent if it has one or infinitely many solutions, and inconsistent if it has no solution.\n\n"
        "Row-Equivalent Augmented Matrices: If two augmented matrices are row equivalent, they have the same solution set.\n\n"
        "A matrix is in row echelon form if all nonzero rows are above any zero rows, each leading entry is to the right of the leading entry of the row above, and entries below a leading entry are zero.\n\n"
        "A matrix is in reduced row echelon form if, in addition, the leading entry in each nonzero row is 1 and is the only nonzero entry in its column.\n\n"
        "Theorem: Uniqueness of the Reduced Echelon Form – Each matrix is row equivalent to one and only one reduced echelon matrix.\n\n"
        "Existence and Uniqueness Theorem: A system is consistent if and only if the rightmost column of its augmented matrix is not a pivot column; if consistent, the solution is unique (no free variables) or infinite (at least one free variable)."
    ),
    "Vector Equations & Span": (
        "Vector Equations and Span:\n"
        "A vector equation of the form x₁a₁ + x₂a₂ + … + xₙaₙ = b has the same solution set as the corresponding linear system. "
        "In particular, b can be generated as a linear combination of a₁, a₂, …, aₙ if and only if a solution exists.\n\n"
        "Span: The span of a set {v₁, …, vₚ} is the collection of all linear combinations c₁v₁ + c₂v₂ + … + cₚvₚ, where the coefficients are scalars."
    ),
    "Matrix Equations & Solutions": (
        "Matrix Equations and Solutions:\n"
        "A matrix equation Ax = b represents a system where A is an m×n matrix, x is an n×1 vector, and b is an m×1 vector. "
        "If A = [a₁, a₂, …, aₙ] and x has entries x₁, …, xₙ, then Ax = x₁a₁ + … + xₙaₙ = b.\n\n"
        "The column space C(A) is defined as {Ax : x ∈ ℝⁿ}. A solution exists if and only if b ∈ C(A).\n\n"
        "Theorem: The matrix equation Ax = b has the same solution set as the vector equation.\n"
        "Theorem: A solution exists if and only if b is in the column space of A.\n"
        "Theorem: If Ax = b has a unique solution, then the homogeneous equation Ax = 0 has only the trivial solution.\n\n"
        "Homogeneous Equation: Ax = 0 has a nontrivial solution if and only if there is at least one free variable.\n"
        "Theorem: For a consistent system Ax = b with particular solution p, every solution is of the form w = p + vₕ, where vₕ is any solution of Ax = 0."
    ),
    "Linear Independence": (
        "Linear Independence:\n"
        "A set of vectors {v₁, …, vₚ} in ℝⁿ is linearly independent if the equation c₁v₁ + … + cₚvₚ = 0 has only the trivial solution (c₁ = c₂ = … = cₚ = 0). "
        "If a nontrivial solution exists, the set is linearly dependent.\n\n"
        "Additional Remarks: A set containing the zero vector is automatically dependent, and if any vector in the set can be expressed as a linear combination of the others, then the set is dependent.\n\n"
        "Theorem: The set is linearly independent if and only if the only solution to c₁v₁ + … + cₚvₚ = 0 is trivial.\n"
        "Theorem: The vectors are independent if and only if the matrix with these vectors as columns has a pivot in every column.\n"
        "Theorem: If S is an independent set and w is not in the span of S, then S ∪ {w} is independent."
    ),
    "Linear Transformations & Matrix Representation": (
        "Linear Transformations & Matrix Representations:\n"
        "A transformation T is linear if T(u + v) = T(u) + T(v) and T(cu) = cT(u) for all u, v and scalars c, which implies T(0) = 0.\n\n"
        "If T: ℝⁿ → ℝᵐ is linear, there exists a unique m×n matrix A such that T(x) = Ax for all x ∈ ℝⁿ. "
        "This A is known as the standard matrix of T.\n\n"
        "Theorem: Let T be linear and let {e₁, …, eₙ} be the standard basis of ℝⁿ. If A is the matrix whose jᵗʰ column is T(eⱼ), then T(x) = Ax for all x, and A is unique.\n"
        "Theorem: T is one-to-one if and only if the equation Ax = 0 has only the trivial solution (i.e., the columns of A are independent).\n"
        "Theorem: T is onto if and only if every b in ℝᵐ can be written as Ax (i.e., the columns of A span ℝᵐ)."
    ),
    "Matrix Operations": (
        "Matrix Operations:\n"
        "Definition of Product: If A is an m×n matrix and B is an n×p matrix, then AB is an m×p matrix whose jᵗʰ column is Abⱼ (i.e., the linear combination of the columns of A using the entries of bⱼ).\n\n"
        "Theorem (Matrix Addition & Scalar Multiplication):\n"
        "   a. A + B = B + A\n"
        "   b. (A + B) + C = A + (B + C)\n"
        "   c. A + 0 = A\n"
        "   d. r(A + B) = rA + rB\n"
        "   e. (r + s)A = rA + sA\n"
        "   f. r(sA) = (rs)A\n\n"
        "Theorem (Properties of Matrix Multiplication):\n"
        "   1. A(BC) = (AB)C\n"
        "   2. A(B + C) = AB + AC\n"
        "   3. (B + C)A = BA + CA\n"
        "   4. r(AB) = (rA)B = A(rB)\n\n"
        "Theorem (Transpose Properties):\n"
        "   1. (Aᵀ)ᵀ = A\n"
        "   2. (A + B)ᵀ = Aᵀ + Bᵀ\n"
        "   3. (rA)ᵀ = rAᵀ\n"
        "   4. (AB)ᵀ = BᵀAᵀ"
    )
}

corpus = []
for topic, full_text in PROMPT_SECTIONS.items():
    paras = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    for i, para in enumerate(paras):
        doc_id = f"{topic}#{i}"
        corpus.append((doc_id, para))

# Embed each chunk
client = openai
texts = [text for (_id, text) in corpus]
embeddings = []
for i in range(0, len(texts), 100):
    batch = texts[i : i + 100]
    resp = client.embeddings.create(model="text-embedding-ada-002", input=batch)
    embeddings.extend(e.embedding for e in resp.data)

emb_array = np.array(embeddings, dtype="float32")
faiss.normalize_L2(emb_array)

# Build FAISS index
dim = emb_array.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb_array)

# Persist index + metadata
faiss.write_index(index, "rag_index.faiss")
with open("rag_corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)

print(" RAG index built (rag_index.faiss) and metadata saved (rag_corpus.pkl)")