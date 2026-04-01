# ============================================================
# RAG CHATBOT WITH VECTOR DATABASE
# Step 1 - Load text from PDFs
# Step 2 - Chunk the text
# Step 3 - Embed each chunk (convert to numbers)
# Step 4 - Store embeddings in FAISS vector database
# Step 5 - Search vector DB for relevant chunks
# Step 6 - Send chunks + question to AI for answer
# ============================================================

import streamlit as st
import numpy as np
import faiss
import requests
from groq import Groq

# ============================================================
# API Key
# ============================================================
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# ============================================================
# STEP 1 - YOUR PDF DATA
# ============================================================
workout_text = """
COMPLETE WORKOUT GUIDE

Introduction to Fitness Goals
Fitness training should always be based on a clear goal such as weight loss, weight gain, or muscle gain. Each goal requires a different approach in terms of exercises, intensity, and frequency. Without a clear goal, workouts become random and results will be slow or inconsistent. A structured plan helps in tracking progress and staying motivated over time.

Types of Workout Goals
Weight Loss: Weight loss focuses on burning calories and reducing body fat. This involves a combination of cardio exercises and light to moderate weight training. Higher repetitions with shorter rest periods help in increasing calorie burn and improving endurance.
Weight Gain: Weight gain focuses on increasing overall body mass. This requires strength training with moderate repetitions and slightly heavier weights. The goal is to stimulate muscle growth while also maintaining a calorie surplus.
Muscle Gain: Muscle gain, also called hypertrophy, focuses on building muscle size and strength. This involves lifting heavier weights with controlled movements. Rest periods are slightly longer to allow muscles to recover between sets.

Workout Split
Workout split refers to how you divide your workouts across the week. Beginners can start with full-body workouts three times a week, which helps in building a strong foundation. Intermediate individuals can follow upper-body and lower-body splits to train more effectively. Advanced individuals often use push, pull, and leg splits to target specific muscle groups and maximize growth.

Training Intensity
Training intensity refers to how hard your workout is. It is usually based on the percentage of maximum weight you can lift. Beginners should start with lighter weights and focus on proper form. As strength improves, weights should be increased gradually. This concept is known as progressive overload and is essential for muscle growth and strength improvement.

Strength Training
Strength training involves lifting weights to build muscle and improve strength. It includes compound exercises like squats, deadlifts, and bench press, which target multiple muscles at once. It also includes isolation exercises like bicep curls and tricep extensions, which focus on specific muscles. Proper form and controlled movements are very important to avoid injuries and maximize results.

Cardio Training
Cardio exercises improve heart health and help in burning calories. There are different types of cardio such as walking, jogging, cycling, and high-intensity interval training HIIT. For weight loss, cardio should be done frequently. For muscle gain or weight gain, cardio should be limited so that it does not interfere with muscle recovery.

Weekly Workout Plan
A weekly workout plan should be balanced and realistic. For weight loss, 5 to 6 workout days including cardio and weights are recommended. For muscle gain, 4 to 5 days of strength training with limited cardio works best. For weight gain, focus more on strength training with minimal cardio. Consistency is more important than doing very intense workouts occasionally.

Rest and Recovery
Rest and recovery are essential parts of any fitness plan. Muscles grow when you rest, not when you are working out. You should aim for 7 to 9 hours of sleep daily. At least one or two rest days per week are necessary to allow the body to recover and prevent injuries.

Progression Strategy
Progression means gradually increasing the difficulty of your workouts. This can be done by increasing weights, repetitions, or sets over time. Tracking your workouts helps you understand your progress and keeps you motivated. Without progression, the body adapts and results will stop.

Common Mistakes
Common mistakes include skipping warm-ups, using improper form, overtraining, and not being consistent. Avoiding these mistakes can significantly improve your results and reduce the risk of injuries.
"""

diet_text = """
COMPLETE DIET AND NUTRITION GUIDE

Introduction to Nutrition
Nutrition is the foundation of fitness. No matter how hard you train, without a proper diet you will not see good results. A balanced diet provides energy, supports recovery, and helps in achieving your fitness goals effectively.

Macronutrients
Macronutrients are nutrients required in large amounts. Protein helps in muscle repair and growth. Carbohydrates provide energy for daily activities and workouts. Fats support hormone function and overall health. A proper balance of all three is necessary for optimal performance.

Micronutrients
Micronutrients include vitamins and minerals that are required in smaller amounts but are equally important. They help in immunity, recovery, and overall body functions. Fruits and vegetables are the best sources of micronutrients.

Diet for Weight Loss
Weight loss requires a calorie deficit, which means consuming fewer calories than you burn. Focus on high-protein foods, reduce sugar intake, and avoid processed foods. Eating whole foods and maintaining portion control is key.

Diet for Weight Gain
Weight gain requires a calorie surplus, meaning you consume more calories than you burn. Include calorie-dense foods like rice, nuts, milk, and healthy fats. Eating frequent meals throughout the day helps in increasing calorie intake.

Diet for Muscle Gain
Muscle gain requires a balanced diet with high protein, moderate carbohydrates, and healthy fats. Post-workout nutrition is very important as it helps in muscle recovery and growth.

Protein Intake
Protein intake depends on your body weight and fitness goals. For muscle gain, you should consume around 1.6 to 2.2 grams of protein per kilogram of body weight. Good sources include eggs, chicken, paneer, lentils, and protein supplements.

Fiber and Digestion
Fiber is important for digestion and gut health. It helps prevent constipation and improves nutrient absorption. Include fruits, vegetables, and whole grains in your daily diet.

Hydration
Water is essential for all body functions. Staying hydrated improves performance, digestion, and recovery. You should aim to drink at least 3 to 4 liters of water daily.

Sample Diet Plans
For weight loss, include low-calorie and high-protein meals such as oats, vegetables, and lean protein. For weight gain, include calorie-rich meals like rice, eggs, milk, and nuts. For muscle gain, include protein in every meal and maintain a balanced intake of all nutrients.

Common Diet Mistakes
Common mistakes include skipping meals, not consuming enough protein, eating too much junk food, and not tracking calorie intake. Avoiding these mistakes will help you achieve your fitness goals faster and more effectively.
"""

# Combine both PDFs
full_text = workout_text + "\n\n" + diet_text

# ============================================================
# STEP 2 - CHUNK: Split text into small pieces
# Why: AI works better with small focused chunks
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
# STEP 3 - EMBED: Convert each chunk into numbers (vectors)
# Why: So computer can understand meaning and do similarity search
# We use a free open source model from HuggingFace API
# ============================================================
def get_embeddings(texts):
API_URL = "https://router.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Content-Type": "application/json"}
    
    all_embeddings = []
    for text in texts:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}}
        )
        result = response.json()
        
        # Handle all possible response formats from HuggingFace
        if isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], list):
                # Format: [[0.1, 0.2, ...]] → take first item
                embedding = result[0]
                if isinstance(embedding[0], list):
                    # Format: [[[0.1, 0.2, ...]]] → average across tokens
                    embedding = np.mean(embedding, axis=0).tolist()
            else:
                # Format: [0.1, 0.2, ...] → use directly
                embedding = result
        elif isinstance(result, dict) and "error" in result:
            # API returned an error - show it clearly
            st.error(f"HuggingFace API error: {result['error']}")
            st.stop()
        else:
            st.error(f"Unexpected response: {result}")
            st.stop()
            
        all_embeddings.append(embedding)
    
    return np.array(all_embeddings, dtype="float32")

# ============================================================
# STEP 4 - VECTOR DATABASE: Store embeddings in FAISS
# Why: FAISS lets us search millions of vectors super fast
# This is the actual vector database storing all chunk vectors
# ============================================================
def build_vector_db(embeddings):
    dimension = embeddings.shape[1]
    # IndexFlatL2 = searches by smallest distance (most similar)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# ============================================================
# STEP 5 - SEARCH: Find most relevant chunks from vector DB
# Why: We only send the most relevant chunks to AI, not everything
# ============================================================
def search_vector_db(question, index, chunks, top_k=3):
    # Embed the question using same model
    question_embedding = get_embeddings([question])
    # Search vector DB for closest chunks
    distances, indices = index.search(question_embedding, top_k)
    # Return the actual text of those chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# ============================================================
# STEP 6 - RAG PIPELINE: Send question + chunks to AI
# RAG = Retrieval Augmented Generation
# Retrieval  = get relevant chunks from vector DB
# Augmented  = add those chunks to the prompt
# Generation = AI generates answer using those chunks
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
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(page_title="Fitness Chatbot", page_icon="💪")
st.title("💪 Fitness & Diet Chatbot")
st.caption("Powered by RAG pipeline with FAISS vector database")

@st.cache_resource
def setup():
    st.write("📄 Step 1 - Loading PDF text...")
    st.write(f"✅ Loaded {len(full_text)} characters")

    st.write("✂️ Step 2 - Chunking text...")
    chunks = chunk_text(full_text)
    st.write(f"✅ Created {len(chunks)} chunks")

    st.write("🔢 Step 3 - Embedding chunks into vectors...")
    embeddings = get_embeddings(chunks)
    st.write(f"✅ Each chunk converted to {embeddings.shape[1]} dimensional vector")

    st.write("🗄️ Step 4 - Storing vectors in FAISS vector database...")
    index = build_vector_db(embeddings)
    st.write(f"✅ Vector database built with {index.ntotal} vectors stored")

    st.write("🚀 RAG pipeline ready!")
    return chunks, index

chunks, index = setup()

# Chat history
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
        with st.spinner("Searching vector database..."):
            # Step 5 - Search vector DB
            relevant_chunks = search_vector_db(user_question, index, chunks)
            # Step 6 - Get AI answer
            answer = ask_ai(user_question, relevant_chunks)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
