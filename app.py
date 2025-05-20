import streamlit as st
import json
import os
from llm_agent import get_conversational_agent
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai._exceptions import RateLimitError
import tiktoken
from bs4 import BeautifulSoup
import requests
import warnings
from bs4 import MarkupResemblesLocatorWarning
from googleapiclient.discovery import build
from google.oauth2 import service_account
import re

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

st.set_page_config(page_title="JobSherpa - AI CV Optimizer", layout="wide")
st.title("📄 AI-Powered CV Optimizer")

# Load available CVs from 'cvs' directory
CV_FOLDER = "cvs"
cv_files = [f for f in os.listdir(CV_FOLDER) if f.endswith(".json")]

# Dropdown to select CV
selected_cv_file = st.selectbox("Select a CV to optimize:", cv_files)

# Load selected CV
cv_path = os.path.join(CV_FOLDER, selected_cv_file)
with open(cv_path, "r", encoding="utf-8") as f:
    cv_data = json.load(f)

# View CV
with st.expander("📄 View CV"):
    st.code(json.dumps(cv_data, indent=2), language="json")

# Job URL input
st.subheader("Job Role URL to Tailor For")
job_url_input = st.text_area("Paste the LinkedIn job description URL:", value=st.session_state.get("job_desc", ""))

def extract_about_this_job_from_url(url: str) -> str:
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        description_div = soup.find('div', class_='description__text description__text--rich')
        if description_div:
            return description_div.get_text(separator='\n', strip=True)
        else:
            return resp.text.strip()
    except Exception as e:
        return f"Error fetching or parsing job description: {e}"

if job_url_input != st.session_state.get("job_desc", ""):
    st.session_state.job_desc = job_url_input
    st.session_state.job_description_text = extract_about_this_job_from_url(job_url_input)
    st.session_state.prompt = None

def create_prompt(cv_json, job_description_text):
    return f"""

You are a career advisor and ResumeMasterGPT, an expert agent trained to generate tailored cv content 

Below is a user's CV in JSON format. Rewrite and enhance this CV based on the job description:

 Output Scope
The main goal is to edit the contents of the existing cv (JSON) based on the job description and instructions given below:

Content Preservation Principles
(1) Resume content accuracy is absolute. Every bullet must faithfully reflect the user's validated experience.
--No inventions.
--No extrapolations.
--No compressions.
--No omissions.
(2) Keyword optimization is secondary.
--Apply keyword phrasing tweaks only if they do not alter the original facts, metrics, or nuance.
(3) When in conflict, prioritize authenticity.
It is better to slightly underuse a keyword than to misrepresent experience.
(4) Please keep the words in UK English for example Organisation rather than organization

Metric and Detail Preservation
-Always preserve key metrics and impact figures in achievements and throughout the document  (e.g. increased sales by 15%)

-the final resume version must be complete, standalone, and fully persuasive.
-Never assume the reviewer has seen prior roles.
-Always include full detail on responsibilities, metrics, and impact — even if similar themes exist across different sections.
-Never compress, omit, or summarize achievements to avoid repetition across versions.

Invention Ban
Never invent roles, achievements, initiatives, projects, frameworks, partnerships, metrics, or results.
Never attribute experience that does not exist in the source resume.
Never create fictionalized enablement initiatives, AI integrations, or LMS deployments unless explicitly confirmed by the user.

Phase 2 — Analysis
(1)
Perform a recruiter mindset scan based on the job description. Emulate how a recruiter or hiring manager would evaluate the resume in 6–8 seconds.
Break this down into three subsections. Identify 4–6 scanning priorities using direct language from the job description.

What the Reviewer is Scanning For:
Identify 4–6 top attributes, responsibilities, or signals that appear repeatedly or implicitly in the job post.

(2) Perform Resume Strength Mapping:
Map user's existing real experience directly to each scanning priority.

What You Already Bring to the Table:
Map relevant, existing resume experience that aligns directly with those scanning points.

(3) Perform an ATS keyword analysis of the job description. Extract the most relevant and recurring keywords or phrases directly from the job post.
Example output format:
 ATS Keyword Analysis (Job Description)

3. Output:
- in valid json format 

Important:
- Never emojis, or informal commentary

{"Here is the job description: " + job_description_text if job_description_text else ""}

CV JSON:
{json.dumps(cv_json, indent=2)}
"""

if "prompt" not in st.session_state or st.session_state.prompt is None:
    job_description_text = st.session_state.get("job_description_text", "")
    st.session_state.prompt = create_prompt(cv_data, job_description_text)

st.subheader("Edit the prompt to customize your CV optimization")
prompt = st.text_area("Prompt:", value=st.session_state.prompt, height=400)

if prompt != st.session_state.prompt:
    st.session_state.prompt = prompt

def count_tokens(text: str, model_name: str = "gpt-4.1") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to a default encoding if the model name is unrecognized
        encoding = tiktoken.get_encoding("cl100k_base")  # Common fallback encoding
    tokens = encoding.encode(text)
    return len(tokens)

@retry(
    wait=wait_random_exponential(min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError),
)
def call_agent(prompt):
    # Use GPT-4.1 model with extended context
    agent = get_conversational_agent(model_name="gpt-4.1")
    return agent.run(prompt)

# Google Docs Integration
def get_google_docs_service():
    scopes = ['https://www.googleapis.com/auth/documents']
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes)
    service = build('docs', 'v1', credentials=credentials)
    return service

def extract_placeholders(document):
    placeholders = set()
    content = document.get('body').get('content')

    for element in content:
        if 'paragraph' in element:
            elements = element['paragraph'].get('elements', [])
            for elem in elements:
                text_run = elem.get('textRun')
                if text_run:
                    text = text_run.get('content')
                    matches = re.findall(r'{{(.*?)}}', text)
                    placeholders.update(matches)
    return placeholders

def replace_placeholders(service, document_id, cv_data):
    requests = []
    for key, value in cv_data.items():
        requests.append({
            'replaceAllText': {
                'containsText': {
                    'text': f'{{{{{key}}}}}',
                    'matchCase': True
                },
                'replaceText': value
            }
        })
    result = service.documents().batchUpdate(
        documentId=document_id, body={'requests': requests}).execute()
    return result

if st.button("🚀 Align CV"):
    token_count = count_tokens(st.session_state.prompt)
    st.info(f"📝 Prompt token count: **{token_count}**")

    max_tokens = 40000  # GPT-4.1 supports up to ~1 million tokens
    if token_count > max_tokens:
        st.error(f"❌ Your prompt is too long by {token_count - max_tokens} tokens. Please shorten the CV or job description or prompt.")
    else:
        with st.spinner("Calling LLM to optimize your CV JSON..."):
            try:
                result = call_agent(st.session_state.prompt)
                try:
                    parsed = json.loads(result)
                    st.success("✅ Valid JSON returned!")
                    st.code(json.dumps(parsed, indent=2), language="json")

                    # Google Docs Integration
                    service = get_google_docs_service()
                    DOCUMENT_ID = '1gjMpzdLazwSEetjz1mzVJkLVwsdRKs3zZnfM8qg4V74'
                    document = service.documents().get(documentId=DOCUMENT_ID).execute()
                    placeholders = extract_placeholders(document)

                    # Map parsed JSON to placeholders
                    cv_mapping = {key: parsed.get(key, '') for key in placeholders}
                    replace_placeholders(service, DOCUMENT_ID, cv_mapping)
                    st.success("✅ Google Doc updated successfully!")

                except json.JSONDecodeError:
                    st.warning("⚠️ The result isn't valid JSON. Showing raw output:")
                    st.code(result)
            except Exception as e:
                st.error(f"❌ Error from LLM: {e}")
