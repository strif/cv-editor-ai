import streamlit as st
import json
import os
from llm_agent import get_conversational_agent
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type, RetryError
from openai._exceptions import RateLimitError
import tiktoken
from bs4 import BeautifulSoup
import requests
import warnings
from bs4 import MarkupResemblesLocatorWarning
from googleapiclient.discovery import build
from google.oauth2 import service_account
import re
from datetime import datetime

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


def clean_json_string(json_str):
    cleaned = ''.join(ch if ch >= ' ' or ch in '\t\n\r' else ' ' for ch in json_str)
    return cleaned


with open(cv_path, "r", encoding="utf-8") as f:
    raw = f.read()
    cleaned_raw = clean_json_string(raw)
    cv_data = json.loads(cleaned_raw)

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


def get_google_docs_service():
    scopes = ['https://www.googleapis.com/auth/documents']
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes)
    service = build('docs', 'v1', credentials=credentials)
    return service


def get_drive_service():
    scopes = ['https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes)
    drive_service = build('drive', 'v3', credentials=credentials)
    return drive_service


def share_document_with_email(drive_service, file_id, user_email):
    permission = {
        'type': 'user',
        'role': 'writer',
        'emailAddress': user_email
    }
    drive_service.permissions().create(
        fileId=file_id,
        body=permission,
        fields='id'
    ).execute()


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

    if not requests:
        return None

    result = service.documents().batchUpdate(
        documentId=document_id, body={'requests': requests}).execute()
    return result


def create_prompt(cv_json, job_description_text, placeholders_list):
    placeholders_str = ", ".join(placeholders_list)
    return f"""

You are a career advisor and ResumeMasterGPT, an expert agent trained to generate tailored CV content. You will be receiving the users cv in JSON format, a job description and a CV template which you will need to create relevant content for.


Content Preservation Principles
(1) Resume content accuracy is absolute. Every bullet must faithfully reflect the user's validated experience. No inventions or No extrapolations. 
(2) Keyword optimization is secondary.
--Apply keyword phrasing tweaks only if they do not alter the original facts, metrics, or nuance.
(3) When in conflict, prioritize authenticity.
It is better to slightly underuse a keyword than to misrepresent experience.
(4) Please keep the words in UK English for example Organisation rather than organization.

Metric and Detail Preservation
- Always preserve key metrics and impact figures in achievements and throughout the document (e.g. increased sales by 15%).
- The final resume version must be complete, standalone, and fully persuasive.
- Never assume the reviewer has seen prior roles.
- Never compress, omit, or summarize achievements to avoid repetition across versions.

Invention Ban
Never invent roles, achievements, initiatives, projects, frameworks, partnerships, metrics, or results. You will need to base your new cv content based on the JSON cv for the user. Never attribute experience that does not exist in the source resume, unless they explicitly appear into the original json cv

Phase 2 — Analysis
(1) Perform a recruiter mindset scan based on the job description. Emulate how a recruiter or hiring manager would evaluate the resume in 6–8 seconds.
Break this down into three subsections. Identify 4–6 scanning priorities using direct language from the job description. You do not need to return this content, but you will use it to create the content to optimise the final cv

What the Reviewer is Scanning For:
Identify 4–6 top attributes, responsibilities, or signals that appear repeatedly or implicitly in the job post.

(2) Perform Resume Strength Mapping:
Map user's existing real experience directly to each scanning priority.

(3) Perform an ATS keyword analysis of the job description. Extract the most relevant and recurring keywords or phrases directly from the job post.
The target Google Docs template contains placeholders that need to be filled e.g. {{ full_name }} or {{ short_description_1 }}
Your task now is to produce a JSON object where each key matches exactly one of the placeholders from the cv template, and the corresponding value is the tailored text content to replace that placeholder in the document.
If you cannot fill a placeholder with relevant content from the CV or job description, set its value to an empty string.
The output JSON should contain only these keys (no extras) and no additional commentary, emojis, or formatting.

---

{"Here is the job description: " + job_description_text if job_description_text else ""}

CV JSON:

{json.dumps(cv_json, indent=2)}
"""


def count_tokens(text: str, model_name: str = "gpt-4-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


@retry(
    wait=wait_random_exponential(min=5, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RateLimitError),
)
def call_agent(prompt):
    agent = get_conversational_agent(model_name="gpt-4-turbo")
    return agent.run(prompt)


if "prompt" not in st.session_state or st.session_state.prompt is None:
    job_description_text = st.session_state.get("job_description_text", "")
    docs_service = get_google_docs_service()
    TEMPLATE_DOC_ID = '1gjMpzdLazwSEetjz1mzVJkLVwsdRKs3zZnfM8qg4V74'
    document = docs_service.documents().get(documentId=TEMPLATE_DOC_ID).execute()
    placeholders = extract_placeholders(document)
    st.session_state.prompt = create_prompt(cv_data, job_description_text, list(placeholders))

st.subheader("Edit the prompt to customize your CV optimization")
prompt = st.text_area("Prompt:", value=st.session_state.prompt, height=400)

if prompt != st.session_state.prompt:
    st.session_state.prompt = prompt

if st.button("🚀 Align CV"):
    token_count = count_tokens(st.session_state.prompt)
    st.session_state.token_count = token_count
    st.session_state.job_desc_token_count = count_tokens(st.session_state.get("job_description_text", ""))
    st.session_state.cv_json_token_count = count_tokens(json.dumps(cv_data, indent=2))
    st.session_state.prompt_instructions_token_count = (
        st.session_state.token_count
        - st.session_state.job_desc_token_count
        - st.session_state.cv_json_token_count
    )

    max_tokens = 40000
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

                    docs_service = get_google_docs_service()
                    drive_service = get_drive_service()

                    TEMPLATE_DOC_ID = '1gjMpzdLazwSEetjz1mzVJkLVwsdRKs3zZnfM8qg4V74'
                    new_doc = drive_service.files().copy(
                        fileId=TEMPLATE_DOC_ID,
                        body={"name": f"Tailored CV - {selected_cv_file}"}
                    ).execute()

                    new_doc_id = new_doc['id']

                    your_email = "kostantinosv@gmail.com"
                    share_document_with_email(drive_service, new_doc_id, your_email)

                    document = docs_service.documents().get(documentId=new_doc_id).execute()
                    placeholders = extract_placeholders(document)

                    st.write(f"Placeholders in document: {placeholders}")
                    st.write(f"Keys in parsed CV JSON: {list(parsed.keys())}")

                    cv_mapping = {key: parsed.get(key, '') for key in placeholders}
                    replace_placeholders(docs_service, new_doc_id, cv_mapping)

                    st.success("✅ New Google Doc created, shared, and updated successfully!")
                    st.markdown(f"🔗 [View Your Tailored CV](https://docs.google.com/document/d/{new_doc_id}/edit)", unsafe_allow_html=True)

                except json.JSONDecodeError:
                    st.warning("⚠️ The result isn't valid JSON. Showing raw output:")
                    st.code(result)
            except RetryError:
                st.error("❌ OpenAI API is rate-limiting. Please wait a moment and try again.")
            except Exception as e:
                st.error(f"❌ Unexpected error from LLM: {e}")

# Safely define debug variables so they exist even if the button hasn't been clicked
token_count = st.session_state.get("token_count", 0)
job_desc_token_count = st.session_state.get("job_desc_token_count", 0)
cv_json_token_count = st.session_state.get("cv_json_token_count", 0)
prompt_instructions_token_count = st.session_state.get("prompt_instructions_token_count", 0)

with st.sidebar.expander("🧠 Debug Info", expanded=False):
    st.write("### Token Count Breakdown")
    st.write(f"📄 Prompt Total: `{token_count}`")
    st.write(f"📜 Job Description: `{job_desc_token_count}`")
    st.write(f"🧾 CV JSON: `{cv_json_token_count}`")
    st.write(f"📘 Prompt Instructions: `{prompt_instructions_token_count}`")
    st.write("### Timestamp")
    st.write(datetime.utcnow().isoformat() + " UTC")
    st.write("### Prompt Preview")
    st.code(st.session_state.prompt[:1000] + "\n...[truncated]...", language="text")
    st.write("### CV JSON Preview")
    st.code(json.dumps(cv_data, indent=2)[:1000] + "\n...[truncated]...", language="json")
