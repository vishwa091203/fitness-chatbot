import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ============================================================
# API Key from Streamlit Secrets
# ============================================================
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# ============================================================
# YOUR PDF CONTENT
# ============================================================
workout_text = """
Workout Split Guide.
Day 1 is Chest and Triceps. Do Bench Press for 4 sets of 10 reps. Do Tricep Pushdowns for 3 sets of 12 reps.
Day 2 is Back and Biceps. Do Pull Ups for 4 sets of 8 reps. Do Barbell Curls for 3 sets of 10 reps.
Day 3 is Rest Day.
Day 4 is Shoulders. Do Overhead Press for 4 sets of 10 reps. Do Lateral Raises for 3 sets of 15 reps.
Day 5 is Legs. Do Squats for 4 sets of 10 reps. Do Leg Press for 3 sets of 12 reps.
Day 6 and Day 7 are Rest Days. 
"""

diet_text = """Diet Guide.
Breakfast should include Oats with banana and peanut butter.
Lunch should include Chicken with rice and vegetables.
Dinner should include Salmon with sweet potato and broccoli.
Snacks can be Greek yogurt, nuts, or protein shake.
Drink at least 3 litres of water every day.
Daily protein target is 150 grams.
Daily calories target is 2500 calories """

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

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

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
