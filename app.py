
import ollama
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tqdm import tqdm
import os

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Load PDF and split into chunks
def load_pdf(pdf_path, chunk_size=500, chunk_overlap=200):
    print("Loading PDF file...")

    pdf = PdfReader(pdf_path)
    text = ""

    for page in tqdm(pdf.pages, desc="Extracting text"):
        page_text = page.extract_text()
        if page_text:
            text += page_text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    chunks = splitter.split_text(text)

    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# Create embeddings
def create_embeddings(chunks, model_name="llama3.1:latest"):
    embeddings = []

    print("Generating embeddings...")

    for chunk in tqdm(chunks):
       response= ollama.embed(
            model="all-minilm",
            input=chunk
        )

    embeddings.append(response["embeddings"][0])

    print(f"✅ Generated {len(embeddings)} embeddings")
    return embeddings

# Upload to Pinecone
def upload_to_pinecone(chunks, embeddings, index_name="buildingrag", batch_size=50):
    index = pc.Index(index_name)

    vectors = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk{i}",
            "values": emb,
            "metadata": {"text": chunk}
        })

    print(f"⬆️ Upserting {len(vectors)} vectors to Pinecone...")

    for i in tqdm(range(0, len(vectors), batch_size)):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    print("✅ Upload complete")
    return index

# Test retrieval
def test_retrieval(question, index, model_name="all-minilm", top_k=3):

    print(f"\n❓ Question: {question}")

    q_embedding = ollama.embed(
        model="all-minilm",
        input=question
    )

    results = index.query(
        vector=q_embedding["embeddings"][0],
        top_k=top_k,
        include_metadata=True
    )

    context = "\n".join(
        [m["metadata"]["text"] for m in results["matches"]]
    )

    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    response = ollama.generate(
        model="qwen2.5:0.5b",
        prompt=prompt
    )

    answer = response["response"]

    print(f"💡 Answer: {answer}")



# MAIN PROGRAM

pdf_path = "node.pdf"

chunks = load_pdf(pdf_path)

resulting_embeddings = create_embeddings(chunks)

upload_to_pinecone(
    chunks,
    resulting_embeddings,
    index_name="buildingrag"
)

index = pc.Index("buildingrag")

answer = test_retrieval(
    "your input query?",
    index
)