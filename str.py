#test
import streamlit as st
import openai
import numpy as np
import re
import pickle
import faiss

# ----------------- Configuration -----------------
openai.api_key = st.secrets.get("OPENAI_API_KEY")

st.set_page_config(page_title="Linear Algebra TF Generator", layout="wide")


# ----------------- Topics & Prompts -----------------
TOPICS = [
    "Systems & Row Reduction",
    "Vector Equations & Span",
    "Matrix Equations & Solutions",
    "Linear Independence",
    "Linear Transformations & Matrix Representation",
    "Matrix Operations",
    "Matrix Inverse",
    "Determinants",
    "Vector Spaces",
]

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
    ),
    "Matrix Inverse": (
    "2×2 Invertibility & Formula:\n"
    "Let A = [a b; c d]. If ad – bc ≠ 0, then A is invertible with\n"
    "A⁻¹ = 1/(ad – bc) · [d  –b;  –c   a].  If ad – bc = 0, then A is not invertible.\n\n"

    "Unique Solution via Inverse:\n"
    "If A is an invertible n×n matrix, then for each b ∈ ℝⁿ the equation Ax = b has the unique solution x = A⁻¹b.\n\n"

    "Basic Inverse Properties:\n"
    "a. If A is invertible then A⁻¹ is invertible and (A⁻¹)⁻¹ = A.\n"
    "b. If A and B are invertible then so is AB, with (AB)⁻¹ = B⁻¹A⁻¹.\n"
    "c. If A is invertible then Aᵀ is invertible and (Aᵀ)⁻¹ = (A⁻¹)ᵀ.\n\n"

    "Elementary Matrices & Row Operations:\n"
    "Performing an elementary row operation on an m×n matrix A is equivalent to multiplying by an m×m elementary matrix E (so the result is EA).  "
    "Each such E is itself invertible, and its inverse is the elementary matrix that undoes that same operation.\n\n"

    "Row‐Equivalence Criterion:\n"
    "An n×n matrix A is invertible ⇔ A is row‐equivalent to Iₙ.  Any sequence of row operations reducing A to Iₙ will transform Iₙ into A⁻¹.\n\n"

    "The Invertible Matrix Theorem:\n"
    "For A ∈ ℝⁿˣⁿ, the following are all equivalent (either all true or all false):\n"
    "  a. A is invertible.\n"
    "  b. A is row‐equivalent to Iₙ.\n"
    "  c. A has n pivot positions.\n"
    "  d. Ax=0 has only the trivial solution.\n"
    "  e. The columns of A are linearly independent.\n"
    "  f. x ↦ Ax is one‐to‐one.\n"
    "  g. Ax=b has at least one solution for each b ∈ ℝⁿ.\n"
    "  h. The columns of A span ℝⁿ.\n"
    "  i. x ↦ Ax maps ℝⁿ onto ℝⁿ.\n"
    "  j. There exists C with CA = I.\n"
    "  k. There exists D with AD = I.\n"
    "  l. Aᵀ is invertible.\n\n"

    "Inverse of a Linear Transformation:\n"
    "Let T: ℝⁿ → ℝⁿ be linear with standard matrix A.  Then T is invertible ⇔ A is invertible.  "
    "In that case, S(x) = A⁻¹x is the unique inverse transformation."
),
"Determinants": (
        "Determinant by Cofactor Expansion (First Row):\n"
        "For n ≥ 2, det(A) = a₁₁ det A₁₁ – a₁₂ det A₁₂ + … + (–1)¹⁺ⁿ a₁ₙ det A₁ₙ = ∑_{j=1}ⁿ (–1)^{1+j} a₁ⱼ det A₁ⱼ.\n\n"
        "Cofactor Expansion General: \n"
        "For any i, det(A) = a_{i1}C_{i1} + a_{i2}C_{i2} + … + a_{in}C_{in}, and for any j, det(A) = a_{1j}C_{1j} + a_{2j}C_{2j} + … + a_{nj}C_{nj}.\n\n"
        "Triangular Matrices: \n"
        "If A is triangular then det(A) = product of its diagonal entries.\n\n"
        "Row Operation Effects:\n"
        "a. Adding a multiple of one row to another leaves det unchanged.\n"
        "b. Swapping two rows multiplies det by –1.\n"
        "c. Multiplying a row by k multiplies det by k.\n\n"
        "Invertibility & Determinant:\n"
        "A square matrix A is invertible ⇔ det(A) ≠ 0.\n"
        "Transpose Property:\n"
        "det(Aᵀ) = det(A).\n"
        "Multiplicative Property:\n"
        "det(AB) = det(A)·det(B)."
    ),
"Vector Spaces": (
    "Vector Space Definition:\n"
    "A vector space V is a nonempty set of vectors with two operations—addition and scalar multiplication—satisfying the following axioms for all u, v, w in V and all scalars c, d:\n\n"
    "Axioms:\n"
    "1. Closure under addition: u + v ∈ V.\n"
    "2. Commutativity of addition: u + v = v + u.\n"
    "3. Associativity of addition: (u + v) + w = u + (v + w).\n"
    "4. Existence of zero vector: ∃0 ∈ V such that u + 0 = u.\n"
    "5. Existence of additive inverse: ∀u ∈ V, ∃(–u) ∈ V with u + (–u) = 0.\n"
    "6. Closure under scalar multiplication: c u ∈ V.\n"
    "7. Distributivity over vector addition: c (u + v) = c u + c v.\n"
    "8. Distributivity over scalar addition: (c + d) u = c u + d u.\n"
    "9. Associativity of scalar multiplication: c (d u) = (c d) u.\n"
    "10. Identity scalar: 1 u = u.\n"
    "Vector Space Consequences:\n"
    "Consequences of the Axioms:\n"
    "For each u in V and each scalar c:\n\n"
    "0·u = 0  # the zero scalar times any vector is the zero vector\n"
    "c·0 = 0  # any scalar times the zero vector is the zero vector\n"
    "–u = (–1)·u  # the additive inverse of u is scalar multiplication by –1\n"
    "Subspace Definition:\n"
    "A subspace H of a vector space V is a subset H ⊆ V satisfying three properties:\n\n"
    "a. The zero vector of V is in H.\n"
    "b. H is closed under addition: if u, v ∈ H then u + v ∈ H.\n"
    "c. H is closed under scalar multiplication: if u ∈ H and c is any scalar then c·u ∈ H.\n\n"
    "Span as a Subspace:\n"
    "If v₁, …, vₚ are vectors in V, then Span{v₁, …, vₚ} = {c₁v₁ + … + cₚvₚ : cᵢ ∈ ℝ} is itself a subspace of V./n"
    "Basis, Spanning Sets & Row/Column Space Bases:"
    "Basis of a Subspace H:\n"
    "Let H be a subspace of a vector space V. A set B in V is a basis for H if:\n"
    "  • B is linearly independent, and\n"
    "  • Span(B) = H.\n\n"
    "The Spanning-Set Theorem:\n"
    "Let S = {v₁, …, vₚ} be a set in V, and let H = Span(S).\n"
    "  a. If some v_k in S is a linear combination of the others, removing v_k still spans H.\n"
    "  b. If H ≠ {0}, then some subset of S is a basis for H.\n\n"
    "Pivot Columns as a Basis for Col A:\n"
    "The pivot columns of an m×n matrix A form a basis for its column space Col A.\n\n"
    "Nonzero Rows as a Basis for Row A:\n"
    "If A and B are row‐equivalent and B is in echelon form, then the nonzero rows of B form a basis for the row space of A (and of B)."
),




}

# ----------------- Helper Functions -----------------
def reformat_matrix_single(matrix_str):
    rows = re.findall(r'\[([^\]]+)\]', matrix_str)
    return "\n".join(r.strip() for r in rows)

def reformat_matrix(matrix_str):
    if "||" in matrix_str:
        parts = matrix_str.split('||')
        labeled = []
        for i, part in enumerate(parts, 1):
            label = 'Primary Matrix:' if i == 1 else f'Secondary Matrix {i}:'
            labeled.append(f"{label}\n" + reformat_matrix_single(part))
        return '\n\n'.join(labeled)
    return "Matrix:\n" + reformat_matrix_single(matrix_str)

# ----------------- Load RAG Index & Corpus -----------------
# Ensure that build_rag_index.py has created these files
def_idx = faiss.read_index("rag_index.faiss")
with open("rag_corpus.pkl","rb") as f:
    def_corpus = pickle.load(f)           # list of (doc_id, text)

# Textbook RAG
book_idx = faiss.read_index("book_index.faiss")
with open("book_chunks.pkl","rb") as f:
    book_corpus = pickle.load(f)          # list of (chunk_id, text)


# ----------------- Retrieval Function -----------------

def retrieve_definitions(selected_topics, k=5):
    query = "Generate true/false questions on: " + ", ".join(selected_topics)
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[query]
    )
    q_emb = np.array(resp.data[0].embedding, dtype="float32")[None,:]
    faiss.normalize_L2(q_emb)
    _, I = def_idx.search(q_emb, k)
    return [def_corpus[i][1] for i in I[0]]

def retrieve_textbook(selected_topics, k=5):
    query = "Generate true/false questions on: " + ", ".join(selected_topics)
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[query]
    )
    q_emb = np.array(resp.data[0].embedding, dtype="float32")[None,:]
    faiss.normalize_L2(q_emb)
    _, I = book_idx.search(q_emb, k)
    return [book_corpus[i][1] for i in I[0]]

# Verification helpers
def is_in_span(vectors, target):
    A = np.array(vectors, float).T
    b = np.array(target, float)
    try:
        x, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if np.allclose(A.dot(x), b, atol=1e-8):
            return True, x.tolist()
    except:
        pass
    return False, None

def is_linearly_independent(vectors):
    A = np.array(vectors, float).T
    return np.linalg.matrix_rank(A) == len(vectors)

def is_row_echelon(matrix):
    M = np.array(matrix, float)
    m, n = M.shape
    pivot_col = -1
    for i in range(m):
        nz = np.where(abs(M[i]) > 1e-8)[0]
        if len(nz) == 0:
            if i < m-1 and np.any(abs(M[i+1:]) > 1e-8): return False
            break
        if nz[0] <= pivot_col: return False
        pivot_col = nz[0]
        if any(abs(M[j, pivot_col])>1e-8 for j in range(i+1,m)): return False
    return True

def is_reduced_row_echelon(matrix):
    M = np.array(matrix, float)
    if not is_row_echelon(M): return False
    m, n = M.shape
    for i in range(m):
        nz = np.where(abs(M[i])>1e-8)[0]
        if not nz: continue
        pc = nz[0]
        if not np.isclose(M[i, pc], 1, atol=1e-8): return False
        if any(j != i and abs(M[j, pc])>1e-8 for j in range(m)): return False
    return True

# GPT call to generate questions
def get_gpt_questions(selected_topics, num_questions, rag_mode):
    defs = "\n\n".join(PROMPT_SECTIONS[t] for t in selected_topics)

    # choose context based on rag_mode
    if rag_mode == "Definitions RAG":
        contexts = retrieve_definitions(selected_topics, k=max(3, num_questions//2))
    elif rag_mode == "Textbook RAG":
        contexts = retrieve_textbook(selected_topics, k=max(3, num_questions//2))
    else:  # "None"
        contexts = None

    if contexts:
        context_text = "\n\n".join(contexts)
        prompt = f"""
You are a linear algebra instructor. Use *only* the facts in the following context to write True/False prompts.

**MATRIX RULE (at least 1/3):**  
At least one-third of your questions **must** include a 3×3 numeric matrix, **formatted** exactly as `[a,b,c][d,e,f][g,h,i]`.  
Any question without a matrix should end with `Matrix: 0`.


Generate {num_questions} challenging true/false linear algebra questions on: {', '.join(selected_topics)}.

Guidelines:
-- Format each clearly as a True/False statement ('Is the following statement true:').
- At least 50% must combine two or more distinct concepts (e.g. span + invertibility).
- Provide a detailed Chain of thought reasoning explanation clearly stating why the statement is true or false.
- Do NOT refer explicitly to theorem numbers.
- When you explain, you may restate relevant definitions or theorems for clarity, but do not literally say ‘as given in the context.’

Format blocks ending with <<END>>:
Question X: …
Answer: True/False
Explanation: …
Matrix: [a,b,c][d,e,f][g,h,i] or 0
<<END>>

Context:
{context_text}



"""
    else:
        prompt = f"""

Generate {num_questions} challenging true/false linear algebra questions for first-year bachelor students, creatively combining concepts from these topics: {', '.join(selected_topics)}.

Guidelines:
- Format each clearly as a True/False statement ('Is the following statement true:').
- At least 50% must combine two or more distinct concepts (e.g. span + invertibility).
- Provide a detailed Chain of thought reasoning explanation clearly stating why the statement is true or false.
- The question needs to be about a numerical matrix in at least a third of the questions, formatted as [a,b,c][d,e,f][g,h,i]; otherwise, put Matrix: 0.

Format each block ending with `<<END>>`:
Question X: …  
Answer: True/False  
Explanation: …  
Matrix: [a,b,c][d,e,f][g,h,i] or 0  
<<END>>
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're an expert linear algebra instructor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=2000
    )
    content = response.choices[0].message.content.strip()
    # Debug raw response
    questions = []
    # Split into blocks by <<END>> delimiter
    blocks = re.split(r"<<END>>", content)
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        q_match = re.search(r'Question\s*\d+:\s*(.*?)\s*Answer:', blk, re.DOTALL)
        a_match = re.search(r'Answer:\s*(True|False)', blk)
        e_match = re.search(r'Explanation:\s*(.*?)\s*Matrix:', blk, re.DOTALL)
        m_match = re.search(r'Matrix:\s*(.*)$', blk, re.DOTALL)
        if not (q_match and a_match and e_match and m_match):
            continue
        q_text = q_match.group(1).strip()
        ans_text = a_match.group(1).strip()
        exp_text = e_match.group(1).strip()
        mat_text = m_match.group(1).strip()
        # Validate matrix format
        if re.match(r'^\[.*\](?:\s*\[.*\])*$', mat_text):
            matrix_str = mat_text
        else:
            matrix_str = "0"
        questions.append({
            'question': q_text,
            'answer': ans_text.lower() == 'true',
            'explanation': exp_text,
            'matrix': matrix_str
        })
    return questions

# Analyze & verify pipeline
def analyze_questions_for_verification(questions):
    analysis = []
    for i, q in enumerate(questions):
     
        pass  
    return analysis

# placeholder for combining analyze & verify & correction
def process_questions(questions):
    
    return questions

# ----------------- App UI -----------------
if 'questions' not in st.session_state:
    st.session_state.questions = []

st.title("🎲 Linear Algebra True/False Question Generator: https://forms.office.com/Pages/ResponsePage.aspx?id=AUGsYwXdcUe81i4qVCHl-kq0V1OUkKtPiig_TzVDGqpUMlVBVkNDVDE4SzBEVEpLQlRSQ0M1T1BJRS4u")

# Configuration form in main area
with st.form(key='config_form'):
    st.subheader("Select Quiz Settings")
    selected_topics = st.multiselect(
        "Choose Topics (pick at least one)", TOPICS, default=[],
        help="Select topics to include in the quiz."
    )
    num_questions = st.number_input(
        "Number of Questions", min_value=1, max_value=20, value=5,
        help="How many True/False questions to generate."
    )
    rag_mode = st.selectbox(
    "RAG mode",
    ["None", "Definitions RAG", "Textbook RAG"],
    index=0
    )
    generate = st.form_submit_button("Generate Questions")

# On generation: clear previous answers and fetch new
if generate:
    if not selected_topics:
        st.error("Please select at least one topic.")
        st.stop()
    if not openai.api_key:
        st.error("OpenAI API key not found. Please add OPENAI_API_KEY to Streamlit secrets.")
        st.stop()
    # Clear prior user responses
    for key in list(st.session_state.keys()):
        if key.startswith('user_ans_') or key.startswith('checked_'):
            del st.session_state[key]
    # Fetch new questions
    with st.spinner("Generating questions..."):
        st.session_state.questions = get_gpt_questions(
            selected_topics, num_questions, rag_mode
        )

# Display questions from session_state
if st.session_state.questions:
    for idx, q in enumerate(st.session_state.questions, 1):
        st.markdown(f"---\n**Question {idx}:** {q['question']}")
        # Display matrix immediately
        if q['matrix'] != "0" and "[" in q['question']:
            st.text(reformat_matrix(q['matrix']))

        # User answer selection with no default
        user_key = f"user_ans_{idx}"
        options = ["Select an answer", "True", "False"]
        if user_key not in st.session_state:
            st.session_state[user_key] = options[0]
        answer = st.selectbox(
            "Your answer:", options, key=user_key
        )

        # Check Answer button and display logic
        checked_key = f"checked_{idx}"
        if st.button("Check Answer", key=f"btn_{idx}"):
            st.session_state[checked_key] = True
        if st.session_state.get(checked_key, False):
            if answer not in ("True", "False"):
                st.warning("Please choose True or False before checking.")
            else:
                correct = 'True' if q['answer'] else 'False'
                if answer == correct:
                    st.success("Correct!")
                else:
                    st.error("Incorrect.")
                st.info(f"Answer: **{correct}**")
                st.write(q['explanation'])

#
