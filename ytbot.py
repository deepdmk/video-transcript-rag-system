# Import necessary libraries and modules
import gradio as gr
import re  # For regular expression operations
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from ibm_watsonx_ai import Credentials  # For API client and credentials management
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # For managing model parameters
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods  # For defining decoding methods
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings  # For interacting with IBM's LLM and embeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes  # For specifying types of embeddings
from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
from langchain.chains import LLMChain  # For creating chains of operations with LLMs
from langchain.prompts import PromptTemplate  # For defining prompt templates

# Function to extract the video ID from a YouTube URL
def get_video_id(url):    
    # Regular expression pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Function to fetch the transcript of a YouTube video
def get_transcript(url):
    # Extract the video ID from the provided YouTube URL
    video_id = get_video_id(url)

    # Return None if URL is invalid
    if video_id is None:
        return None

    # Initialize the YouTubeTranscriptApi object
    ytt_api = YouTubeTranscriptApi()
    
    # Fetch the transcript list for the video
    transcripts = ytt_api.list(video_id)
    
    transcript = ""
    for t in transcripts:
        # Check if the transcript is in English
        if t.language_code == 'en':
            if t.is_generated:
                #  If an auto-generated transcript is found, use it
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                #  If a manually created transcript is found, use it
                transcript = t.fetch()
                break  # Stop after finding the first manually created transcript
    
    return transcript if transcript else None

# Function to process the fetched transcript
def process(transcript):
    # Initialize an empty string to store the processed transcript
    txt = ""
    
    # Iterate over each entry in the transcript list
    for i in transcript:
        try:
            # Concatenate the text and start time of each transcript entry
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            pass
            
    # Return the processed transcript
    return txt

# Function to split the processed transcript into manageable chunks of 300 and 40 overlapping characters
def chunk_transcript(processed_transcript, chunk_size=300, chunk_overlap=40):
    # Initialize the text splitter with the specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the processed transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks

# Function to set up IBM Watson credentials and project details
def setup_credentials():
    # Define the model ID for the WatsonX model being used
    model_id = "meta-llama/llama-3-2-3b-instruct"
    
    # Initialize the credentials for IBM Watson services
    credentials = Credentials(url='input credentials for Watson Watson services here')
    
    # Define the project ID for the IBM Watson project
    project_id = "input project ID here"
    
    # Return the model ID, credentials, and project ID
    return model_id, credentials, project_id

# Function to define parameters for the WatsonX model
def define_parameters():
    # Return a dictionary containing the parameters for the WatsonX model
    return {
        # Specify the decoding method for the model as greedy
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        
        # Specify the maximum number of new tokens to generate at 900
        GenParams.MAX_NEW_TOKENS: 900,
    }

# Function to initialize the WatsonX LLM with the provided parameters and credentials
def initialize_watsonx_llm(model_id, credentials, project_id, parameters):
    # Create and return an instance of the WatsonxLLM with the specified configuration
    return WatsonxLLM(
        model_id=model_id,          # Set the model ID for the LLM
        url=credentials.url,      # Retrieve the service URL from credentials
        project_id=project_id,            # Set the project ID for accessing resources
        params=parameters                  # Pass the parameters for model behavior
    )


# Function to set up the embedding model for the WatsonX environment
def setup_embedding_model(credentials, project_id):
    # Create and return an instance of the WatsonxEmbeddings with the specified configuration
    return WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,  # Specify the embedding model type
        url=credentials.url,                            
        project_id=project_id                               
    )


# Function to create a FAISS index from text chunks using the specified embedding model
def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.
    
    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :return: FAISS index
    """
    #  
    return FAISS.from_texts(chunks, embedding_model)


# Function to perform a similarity search on the FAISS index
def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index.
    
    :param faiss_index: The FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    :return: List of similar results
    """
    # Perform the similarity search and return the results
    results = faiss_index.similarity_search(query, k=k)
    return results

# Function to create a summary prompt template for the WatsonX LLM
def create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
    
    :return: PromptTemplate object
    """
    # Define the template string for the summary prompt
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    # Create the PromptTemplate object
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )
    
    return prompt

# Function to create an LLMChain for generating summaries
def create_summary_chain(llm, prompt, verbose=True):
    """
    Create an LLMChain for generating summaries.
    
    :param llm: Language model instance
    :param prompt: PromptTemplate instance
    :param verbose: Boolean to enable verbose output (default: True)
    :return: LLMChain instance
    """
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)

# Function to generate a summary of the video using the provided LLMChain
def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 3).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context

# Function to create a question-answering prompt template
def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.
    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """
    
    # Define the template string for the Q&A prompt
    qa_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
    1. Precise and free from repetition
    2. Consistent with the information provided in the video
    3. Well-organized and easy to understand
    4. Focused on addressing the user's question directly
    If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.
    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    # Create the PromptTemplate object
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template

# Function to create an LLMChain for question answering
def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
            The language model to use in the chain (e.g., WatsonxGranite).
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.
        verbose: bool, optional (default=True)
            Whether to enable verbose output for the chain.

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """
    
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)

# Function to generate an answer based on user input
def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.

    Returns:
        str: The generated answer to the user's question.
    """

    # Retrieve relevant context from the FAISS index
    relevant_context = retrieve(question, faiss_index, k=k)

    # Generate the answer using the Q&A chain
    answer = qa_chain.predict(context=relevant_context, question=question)

    return answer


# Global variables to store transcript state between function calls
fetched_transcript = None
processed_transcript = ""

# Function to summarize the video based on the fetched and processed transcript
def summarize_video(video_url):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript
    
    # Check if the transcript needs to be fetched
    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    # Check if the transcript is available for summarization
    if processed_transcript:
        # Step 1: Set up IBM Watson credentials
        model_id, credentials, project_id = setup_credentials()

        # Step 2: Initialize WatsonX LLM for summarization
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

        # Step 3: Create the summary prompt and chain
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        # Step 4: Generate the video summary
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."

# Function to answer a user's question based on the fetched and processed transcript
def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global fetched_transcript, processed_transcript

    # Check if the transcript needs to be fetched
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    # Check if the transcript is available for Q&A
    if processed_transcript and user_question:
        # Step 1: Chunk the transcript (only for Q&A)
        chunks = chunk_transcript(processed_transcript)

        # Step 2: Set up IBM Watson credentials
        model_id, credentials, project_id = setup_credentials()

        # Step 3: Initialize WatsonX LLM for Q&A
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

        # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)
        embedding_model = setup_embedding_model(credentials, project_id)
        faiss_index = create_faiss_index(chunks, embedding_model)

        # Step 5: Set up the Q&A prompt and chain
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        # Step 6: Generate the answer using FAISS index
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."


# Create a Gradio interface for user interaction
with gr.Blocks() as interface:

    gr.Markdown(
        "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
    )

    # Input for YouTube video URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")
    
    # Output for video summary and Q&A
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    # Buttons for summarizing the video and asking questions
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    # Optional: Display transcript status
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    # Define button click actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

# Launch the Gradio interface
interface.launch(server_name="0.0.0.0", server_port=7860)