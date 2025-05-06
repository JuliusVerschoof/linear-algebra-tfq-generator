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
    "Matrix Operations"
]

PROMPT_SECTIONS = {
    "Systems & Row Reduction": (
        "System of Linear Equations & Row Reduction:\n"
        "A system of linear equations has: no solution, or exactly one solution, or infinitely many solutions. "
        "A system is consistent if it has one or infinitely many solutions, and inconsistent if it has no solution.\n\n"
        "Row-Equivalent Augmented Matrices: If two augmented matrices are row equivalent, they have the same solution set.\n\n"
        "A matrix is in row echelon form if all nonzero rows are above any zero rows, each leading entry is to the right of the leading entry of the row above, and entries below a leading entry are zero.\n\n"
        "A matrix is in reduced row echelon form if, in addition, the leading entry in each nonzero row is 1 and is the only nonzero entry in its column.\n\n"
        "Theorem 1: Uniqueness of the Reduced Echelon Form – Each matrix is row equivalent to one and only one reduced echelon matrix.\n\n"
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
        "Theorem 3: The matrix equation Ax = b has the same solution set as the vector equation.\n"
        "Theorem 4: A solution exists if and only if b is in the column space of A.\n"
        "Theorem 5: If Ax = b has a unique solution, then the homogeneous equation Ax = 0 has only the trivial solution.\n\n"
        "Homogeneous Equation: Ax = 0 has a nontrivial solution if and only if there is at least one free variable.\n"
        "Theorem 6: For a consistent system Ax = b with particular solution p, every solution is of the form w = p + vₕ, where vₕ is any solution of Ax = 0."
    ),
    "Linear Independence": (
        "Linear Independence:\n"
        "A set of vectors {v₁, …, vₚ} in ℝⁿ is linearly independent if the equation c₁v₁ + … + cₚvₚ = 0 has only the trivial solution (c₁ = c₂ = … = cₚ = 0). "
        "If a nontrivial solution exists, the set is linearly dependent.\n\n"
        "Additional Remarks: A set containing the zero vector is automatically dependent, and if any vector in the set can be expressed as a linear combination of the others, then the set is dependent.\n\n"
        "Theorem 7: The set is linearly independent if and only if the only solution to c₁v₁ + … + cₚvₚ = 0 is trivial.\n"
        "Theorem 8: The vectors are independent if and only if the matrix with these vectors as columns has a pivot in every column.\n"
        "Theorem 9: If S is an independent set and w is not in the span of S, then S ∪ {w} is independent."
    ),
    "Linear Transformations & Matrix Representation": (
        "Linear Transformations & Matrix Representations:\n"
        "A transformation T is linear if T(u + v) = T(u) + T(v) and T(cu) = cT(u) for all u, v and scalars c, which implies T(0) = 0.\n\n"
        "If T: ℝⁿ → ℝᵐ is linear, there exists a unique m×n matrix A such that T(x) = Ax for all x ∈ ℝⁿ. "
        "This A is known as the standard matrix of T.\n\n"
        "Theorem 10: Let T be linear and let {e₁, …, eₙ} be the standard basis of ℝⁿ. If A is the matrix whose jᵗʰ column is T(eⱼ), then T(x) = Ax for all x, and A is unique.\n"
        "Theorem 11: T is one-to-one if and only if the equation Ax = 0 has only the trivial solution (i.e., the columns of A are independent).\n"
        "Theorem 12: T is onto if and only if every b in ℝᵐ can be written as Ax (i.e., the columns of A span ℝᵐ)."
    ),
    "Matrix Operations": (
        "Matrix Operations:\n"
        "Definition of Product: If A is an m×n matrix and B is an n×p matrix, then AB is an m×p matrix whose jᵗʰ column is Abⱼ (i.e., the linear combination of the columns of A using the entries of bⱼ).\n\n"
        "Theorem 1 (Matrix Addition & Scalar Multiplication):\n"
        "   a. A + B = B + A\n"
        "   b. (A + B) + C = A + (B + C)\n"
        "   c. A + 0 = A\n"
        "   d. r(A + B) = rA + rB\n"
        "   e. (r + s)A = rA + sA\n"
        "   f. r(sA) = (rs)A\n\n"
        "Theorem 2 (Properties of Matrix Multiplication):\n"
        "   1. A(BC) = (AB)C\n"
        "   2. A(B + C) = AB + AC\n"
        "   3. (B + C)A = BA + CA\n"
        "   4. r(AB) = (rA)B = A(rB)\n\n"
        "Theorem 3 (Transpose Properties):\n"
        "   1. (Aᵀ)ᵀ = A\n"
        "   2. (A + B)ᵀ = Aᵀ + Bᵀ\n"
        "   3. (rA)ᵀ = rAᵀ\n"
        "   4. (AB)ᵀ = BᵀAᵀ"
    )
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

Context:
{context_text}


Generate {num_questions} challenging true/false linear algebra questions on: {', '.join(selected_topics)}.

Guidelines:
- At least one-third must include a specific 3×3 numeric matrix; others use 'Matrix: 0'.
- Use a Detailed chain of thought explanation stating why each is true or false.

Format blocks ending with <<END>>:
Question X: …
Answer: True/False
Explanation: …
Matrix: [a,b,c][d,e,f][g,h,i] or 0
<<END>>
"""
    else:
        prompt = f"""

Generate {num_questions} challenging true/false linear algebra questions for first-year bachelor students, creatively combining concepts from these topics: {', '.join(selected_topics)}.

Guidelines:
- Format each clearly as a True/False statement ('Is the following statement true:').
- At least 50% must combine two or more distinct concepts (e.g. span + invertibility).
- Provide a detailed Chain of thought reasoning explanation clearly stating why the statement is true or false.
- Do NOT refer explicitly to theorem numbers.
- When you explain, you may restate relevant definitions or theorems for clarity, but do not literally say ‘as given in the context.’
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
        q_text = q_match.group(1).strip().strip('"')
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

st.title("🎲 Linear Algebra True/False Question Generator")

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
            st.markdown("**Matrix:**")
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
