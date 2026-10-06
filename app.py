import streamlit as st
import rag

# Streamlit runs this whole file from top to bottom every time you click something.

st.title("📄 مرشد الأبحاث - Arabic Paper Helper")

# ---------------- Part 1: upload and index PDFs ----------------
st.header("1) Upload papers")

files = st.file_uploader("Choose English PDF files", type="pdf", accept_multiple_files=True)

if st.button("Index"):                       # runs only when the button is clicked
    for f in files:
        path = rag.UPLOAD_DIR / f.name       # where to save the file
        path.write_bytes(f.getbuffer())      # save it
        n = rag.index_pdf(path)              # parse -> split -> embed -> FAISS
        st.success(f"{f.name}: {n} chunks")

st.write("Indexed papers:", rag.list_docs())

# ---------------- Part 2: ask a question ----------------
st.header("2) Ask in Arabic")

question = st.text_input("اكتب سؤالك هنا")

if st.button("Ask") and question:
    result = rag.ask(question)               # retrieve + Qwen answer

    st.write(result["answer"])
    st.write("Confidence:", result["confidence"], "%")
    st.write("Category:", result["category"])
    st.write("Follow-up:", result["follow_up"])

    st.subheader("Sources")
    for doc, similarity in result["hits"]:
        st.write(f"{doc.metadata['doc']} - page {doc.metadata['page'] + 1} - similarity {similarity:.2f}")
        st.write(doc.page_content)