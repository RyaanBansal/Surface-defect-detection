import base64
import os
from groq import Groq
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

print("GROQ API Key Loaded:", bool(groq_api_key))

def save_responses_to_pdf(responses, output_path="surface-defects.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    for idx, text in enumerate(responses):
        title = Paragraph(f"<b>🖼️ Image {idx + 1} Result:</b>", styles["Heading4"])
        paragraph = Paragraph(text.replace("\n", "<br />"), styles["Normal"])
        story.extend([title, Spacer(1, 0.2*cm), paragraph, Spacer(1, 1*cm)])

    doc.build(story)

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_base64_images(folder_path):
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
    
    base64_images = []
    for file in files:
        path = os.path.join(folder_path, file)
        encoded = encode_image(path)
        base64_images.append(encoded)
    
    return base64_images

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.2)

def rag_pipeline(query: str, string_data: str):
    print("🔹 Converting strings into LangChain documents...")
    documents = Document(page_content=string_data)
    print(f"✅ Loaded document.")

    print("🔹 Splitting text into chunks...")
    chunks = text_splitter.split_documents([documents])
    print(f"✅ Created {len(chunks)} text chunks.")

    try:
        print("🔹 Creating FAISS vector store...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("✅ FAISS vector store created.")
    except Exception as e:
        print("❌ Error creating FAISS vector store:", e)
        return "Error in vector store creation."

    print("🔹 Retrieving relevant documents...")
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(query)
    print(f"✅ Retrieved {len(retrieved_docs)} relevant document(s).")

    if not retrieved_docs:
        print("⚠️ No relevant documents found.")
        return "No relevant information found."

    print("🔹 Generating answer using Groq LLM...")
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    You are an expert at analysing different types of surfaces and are capable of determining the existence of defects.
    Below is a paragraph describing a surface, which may or may not have defects.
    Analyze the text thoroughly and answer the question.

    Context:
    {context}

    Question:
    {query}

    Present the answer as a single line.
    """
    answer = llm.invoke(prompt)
    print("✅ Answer generated.")
    return getattr(answer, "content", answer)

question = "For the given image description, label the surface as 'defective' and 'non-defective'. The surface is defective if it contains any major defects such sa big cracks."

# Set your Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Path to your image
#folder = "Sample set-surfaces"
folder = "Sample set-marble"

prompt_text = "Given the image of a surface, describe the surface and highlight any major cracks visible. Create the responses in a paragraph form."

all_response=[]
batch_size = 10
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
files = sorted([f for f in os.listdir(folder) if f.lower().endswith(valid_exts)])


#base64_images = get_base64_images(folder)

for batch_start in range(0, len(files), batch_size):
    batch_files = files[batch_start:batch_start+batch_size]
    print(f"processing batch {batch_start//batch_size+1} with {len(batch_files)} images")
    # Send request to Groq LLM with image and prompt
    for idx, file_name in enumerate(batch_files, start = batch_start):
        file_path = os.path.join(folder, file_name)
        try:
            encoded_image = encode_image(file_path)
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}",
                                    "detail": "auto"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.5,
                max_tokens=2048,
                top_p=1,
                stream=False
            )
            result = response.choices[0].message.content
            print(f"image [{idx+1}]: ", result)
            answer = rag_pipeline(question, result)
            print(f"\n Answer: {answer}")
            combine = result + "\n\n<b>image label:</b> " + answer
            all_response.append(combine)
        except Exception as e:
            print(f"❌ Error on image [{idx}]: {e}")

save_responses_to_pdf(all_response)
