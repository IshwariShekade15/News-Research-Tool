import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # Add this line
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv 
load_dotenv()
import os
from  qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore as Qdrant
from langchain.chains import RetrievalQAWithSourcesChain
import streamlit as st
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #E3F2FD !important;
    }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-13ln4jf {
        background-color: #E3F2FD !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def init_qdrant_client():
    api_key = st.secrets.get("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")
    url = st.secrets.get("QDRANT_URL") or os.getenv("QDRANT_URL")
    timeout = 300

    if api_key and url:
        return QdrantClient(
            api_key=api_key,
            url=url,
            timeout=timeout
        )
    else:
        return QdrantClient(":memory:", timeout=timeout)
    
#collection name for qdrant 
collection_name = "demo1"

#client = init_qdrant_client()
#if client.collection_exists(collection_name):
#    client.delete_collection(collection_name)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

st.title("News Research Tool 📰")

# Sidebar for URLs
st.sidebar.title("Enter URLs:")
urls = []
for i in range(3):
    urls.append(st.sidebar.text_input(f"URL No.{i+1}"))

# Process URLs button
process_btn = st.sidebar.button("🔄 Process URLs")



def has_valid_data():
    """Check if qdrant exists and has data """
    try:
        client = init_qdrant_client()
        if client.collection_exists(collection_name):
            collection_info = client.get_collection(collection_name)
            return collection_info.points_count > 0
        return False
    except:
        return False
    

if has_valid_data():
    st.sidebar.success(" URLs processed! Enhanced answers available.")
    if st.sidebar.button(" Clear processed data"):
        
        try:
            client = init_qdrant_client()
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
            st.sidebar.success("🗑️ Data cleared!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error clearing data: {e}")

# Main query section
query = st.text_area("Please Enter your Query:", height=100, placeholder="Ask any question...")

# Get Answer button
if st.button("🔍 Get Answer", type="primary") and query.strip():
    
    # Check if any URLs are provided
    valid_urls = [url.strip() for url in urls if url.strip()]
    has_processed_data = has_valid_data()
    
    # Case 1: No URLs and no processed data - Use LLM only
    if not valid_urls and not has_processed_data:
        st.info("No URLs entered! Providing general answer without reference data...")
        
        try:
            with st.spinner("Getting answer from LLM..."):
                # Direct LLM call without retrieval
                response = llm.invoke(query)
                
            st.header(" Answer (General Knowledge):")
            st.write(response.content)
            st.warning(" **Tip:** Add URLs above for more accurate, source-based answers!")
            
        except Exception as e:
            st.error(f" Error getting answer: {str(e)}")
    
    # Case 2: URLs provided but not processed yet
    elif valid_urls and not has_processed_data:
        st.warning("⚠️ URLs entered but not processed yet. Please click 'Process URLs' first!")
    
    # Case 3: URLs are processed - Use RAG with sources
    elif has_processed_data:
        try:
            with st.spinner("🔍 Searching through processed documents..."):
                # Initialize Qdrant vector store
                client = init_qdrant_client()
                embeddings = load_embeddings()
                
                vector_store = Qdrant(
                    client=client,
                    collection_name=collection_name,
                    embedding=embeddings
                )
                # Create retrieval chain
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, 
                    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
                )
                
                # Get answer with sources
                result = chain.invoke({"question": query}, return_only_outputs=True)
            
            st.header(" Answer (Based on Your URLs):")
            st.write(result["answer"])
            
            # Show sources if available
            if "sources" in result and result["sources"]:
                st.header(" Sources:")
                st.write(result["sources"])
            else:
                st.info(" Answer generated from processed content")
                
        except Exception as e:
            
            st.info("💡 Try processing URLs again or ask a general question")

# Process URLs section
if process_btn:
    valid_urls = [url.strip() for url in urls if url.strip()]
    
    if not valid_urls:
        st.error(" Please enter at least one valid URL to process!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load data from URLs
            status_text.text("Loading data from URLs...")
            progress_bar.progress(20)
            
            loader = UnstructuredURLLoader(urls=valid_urls)
            data = loader.load()
            
            if not data:
                st.error(" No data could be loaded from the URLs. Please check if URLs are valid.")
            else:
                # Step 2: Split text into chunks
                status_text.text(" Splitting text into chunks...")
                progress_bar.progress(40)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n" , "." , " " , ","],
                    chunk_size=150, 
                    chunk_overlap=50  
                )
                
                docs = text_splitter.split_documents(data)
                docs = [doc for doc in docs if len(doc.page_content) <= 600]
                
                
                if not docs:
                    st.error(" No valid text chunks created from the URLs.")

                else:
                    # Step 3: Create embeddings
                    status_text.text("Setting up Qdrant collection....")
                    progress_bar.progress(60)
                    

                    client = init_qdrant_client()
                    embeddings = load_embeddings()
                    
                    # Create collection if it doesn't exist
                    if not client.collection_exists(collection_name):
                        client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=384,  # all-MiniLM-L6-v2 has 384 dimensions
                                distance=Distance.COSINE
                            )
                        )
                    else:
                        # Clear existing collection
                        client.delete_collection(collection_name)
                        client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=384,
                                distance=Distance.COSINE
                            )
                        )
         # Step 4: Create vector store and add documents
                    status_text.text("💾 Adding documents to Qdrant...")
                    progress_bar.progress(80)
                    
                    vector_store = Qdrant(
                        client=client,
                        collection_name=collection_name,
                        embedding=embeddings
                    )

                    vector_store.add_documents(docs)

                    progress_bar.progress(100)
                    status_text.success("✅ URLs processed successfully!")
                    
                    # Show collection info
                    collection_info = client.get_collection(collection_name)
                    st.success(f"🎉 Processed {len(valid_urls)} URLs with {collection_info.points_count} text chunks!")

        except Exception as e:
            st.error(f"❌ Error processing URLs: {str(e)}")
            status_text.error("Processing failed!")
                
