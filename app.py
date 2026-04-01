import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ============================================================
# API Key from Streamlit Secrets
# ============================================================
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# ============================================================
# YOUR PDF CONTENT
# ============================================================
workout_text = """
%PDF-1.7
%����
1 0 obj
<</Type/Catalog/Pages 2 0 R/Lang(en) /StructTreeRoot 33 0 R/MarkInfo<</Marked true>>/Metadata 144 0 R/ViewerPreferences 145 0 R>>
endobj
2 0 obj
<</Type/Pages/Count 3/Kids[ 3 0 R 28 0 R 30 0 R] >>
endobj
3 0 obj
<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 5 0 R/F2 12 0 R/F3 17 0 R/F4 19 0 R/F5 24 0 R/F6 26 0 R>>/ExtGState<</GS10 10 0 R/GS11 11 0 R>>/ProcSet[/PDF/Text/ImageB/ImageC/ImageI] >>/MediaBox[ 0 0 595.32 841.92] /Contents 4 0 R/Group<</Type/Group/S/Transparency/CS/DeviceRGB>>/Tabs/S/StructParents 0>>
endobj
4 0 obj
<</Filter/FlateDecode/Length 3455>>
stream
x��\K�#��0�������
"""

diet_text = """
PASTE YOUR DIET PDF TEXT HERE
"""

full_text = workout_text + "\n\n" + diet_text

# ============================================================
# CHUNK
# ============================================================
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# ============================================================
# BUILD SEARCH INDEX
# ============================================================
def build_vectorizer(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors

# ============================================================
# SEARCH
# ============================================================
def search(question, vectorizer, vectors, chunks, top_k=3):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [chunks[i] for i in top_indices]
    return results

# ============================================================
# ASK AI — using Gemini (free, no credit card needed)
# ============================================================
def ask_ai(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)

    prompt = f"""You are a helpful fitness assistant.
Use ONLY the information below to answer the question.
If the answer is not in the information, say "I don't know based on the provided guides."

INFORMATION FROM PDF:
{context}

USER QUESTION:
{question}

ANSWER:"""

    response = model.generate_content(prompt)
    return response.text

# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(page_title="Fitness Chatbot", page_icon="💪")
st.title("💪 Fitness & Diet Chatbot")
st.caption("Answers based only on your uploaded PDF guides")

@st.cache_resource
def setup():
    st.write("✂️ Chunking text...")
    chunks = chunk_text(full_text)
    st.write(f"✅ Created {len(chunks)} chunks")
    st.write("🔢 Building search index...")
    vectorizer, vectors = build_vectorizer(chunks)
    st.write("✅ Ready!")
    return chunks, vectorizer, vectors

chunks, vectorizer, vectors = setup()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_question := st.chat_input("Ask about your workout or diet..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant_chunks = search(user_question, vectorizer, vectors, chunks)
            answer = ask_ai(user_question, relevant_chunks)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
