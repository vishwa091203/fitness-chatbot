import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

api_key = st.secrets["OPENROUTER_API_KEY"]


client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    timeout=120.0
)

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
%PDF-1.7
%����
1 0 obj
<</Type/Catalog/Pages 2 0 R/Lang(en) /StructTreeRoot 33 0 R/MarkInfo<</Marked true>>/Metadata 152 0 R/ViewerPreferences 153 0 R>>
endobj
2 0 obj
<</Type/Pages/Count 3/Kids[ 3 0 R 28 0 R 30 0 R] >>
endobj
3 0 obj
<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 5 0 R/F2 12 0 R/F3 17 0 R/F4 19 0 R/F5 24 0 R/F6 26 0 R>>/ExtGState<</GS10 10 0 R/GS11 11 0 R>>/ProcSet[/PDF/Text/ImageB/ImageC/ImageI] >>/MediaBox[ 0 0 595.32 841.92] /Contents 4 0 R/Group<</Type/Group/S/Transparency/CS/DeviceRGB>>/Tabs/S/StructParents 0>>
endobj
4 0 obj
<</Filter/FlateDecode/Length 3054>>
stream
x��\K���0�AG�@k�&4�k�����`���H6�����Q����Y`�g$>>?YK�}��߿~��?^~�v�ß�����O�~�v�����������iG:9Ȟ���������߾�>=?�_��>�Ў�����'(J:�i�&:-U?���_�O����\0����J��BI��Б&��RZڟ����Gw�������ߏ������^���t�(����5��텁�O���jDx{<B���,&��������0�j�?
<��ՀА9�������̖C1��f6KV����o2ݪ���f�?b�x��1B�m<H�@������(B�����������pK�mf�u��}{*ASش��j(w��R��
n�dh�h��뇫�ψ���)Mڱ%aL�����m93��T޶�t��ek�"�mE��c,����㼅������x������xt_�����F����hq���#�FK�&F��^�NϸW#��
o��LCy̢R�/�gt���r�u�/e�W��b���Bdl��"�7KT����{�'̊�+���
0�
�`�`���V�Pb�Q
�!_�P%�7T���'?r����i`�aKJ�x��6�v+��F����ÊxY���V
`���
�e�j9\�z�|V�m��!���a��t�uC�P��p�/��\��j�`Gl$"�������m�gԿ�ꔤf赮l��\����b��^P����В�9��8��̑yv3K��^H�z6�6V)���Gi��se�i��ђ֐����l��lg�!tK�f�尐�3���S����mdɿ��Z��x�W��<�<Կ��4�0�����]KԨ�)L$L�&��N
�r���lF�lg�)�[S��,ua[�v�-v����ֶi�-�-i@4ئ��IߖWg+��;�j�*�~U9�
D�������A�m�ZKס�)Klh`IiG��Y�v��e���ٌ����Y�^�c��u�P=��JW@N2=2st�����4��U��a�ݬ��n�R��e�8+��/���܌����=*�Qz��WyKFP��yr�a����㗨iy�J���ݺ�ږ,x�¤��P�MZ����e��ٌ����7ZZ9G��Z�r�>�-�+��BPb��&)�j��I���#�ZW!^v2�IE���cm��|g����=,˜:�I�7�� Yq�
���[a�9[�Hћ���J3�2Lvo��Id��k*��6ci�3L{������ɹE5��D��9O��&�6�ճe�@��T���Z٤%VdC&�+��Sm�͆^h�����f����]l�/Scz]�Q��X0U�	<��!���S���v��B{ۥd��&iK�K{�����J;��'��y,��,�.l�����fw
6rE
6J�v���W*˦���
"""

full_text = workout_text + "\n\n" + diet_text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def build_vectorizer(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors

def search(question, vectorizer, vectors, chunks, top_k=3):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [chunks[i] for i in top_indices]
    return results

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
    response = client.chat.completions.create(
        model="meta-llama/llama-3.2-3b-instruct:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

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
