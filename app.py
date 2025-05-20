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

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

st.set_page_config(page_title="JobSherpa - AI CV Optimizer", layout="wide")
st.title("📄 AI-Powered CV Optimizer")

# === Load CV JSON files from local 'cvs' folder ===
CV_FOLDER = "cvs"
cv_files = {
    os.path.splitext(file)[0].replace("_", " ").title(): os.path.join(CV_FOLDER, file)
    for file in os.listdir(CV_FOLDER) if file.endswith(".json")
}

# Dropdown to select a CV
st.subheader("Select a CV to Optimize")
selected_cv_name = st.selectbox("Candidate", options=list(cv_files.keys()))
cv_path = cv_files[selected_cv_name]

# Load selected CV JSON
with open(cv_path, "r", encoding="utf-8") as f:
    cv_data = json.load(f)

# Show CV JSON (expandable)
with st.expander(f"📄 View CV JSON for {selected_cv_name}"):
    st.code(json.dumps(cv_data, indent=2), language="json")

# === Job Description Input ===
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

st.subheader("Job Role URL to Tailor For")
job_url_input = st.text_area("Paste the LinkedIn job description URL:", value=st.session_state.get("job_desc", ""))

if job_url_input != st.session_state.get("job_desc", ""):
    st.session_state.job_desc = job_url_input
    st.session_state.job_description_text = extract_about_this_job_from_url(job_url_input)
    st.session_state.prompt = None

# === Prompt Creation ===
def create_prompt(cv_json, job_description_text):
    return f"""
You are an expert career advisor helping improve a JSON-based CV.

Below is a user's CV in JSON format. Rewrite and enhance this CV based on the job description:

- Keep it in valid JSON structure.
- Highlight key skills and achievements 
- Pick or add skills that are relevant to the industry and role
- Prioritize clarity, brevity, and impact.
- Do not omit important responsibilities unless redundant.
- Create a profile description that is suitable for the job description, e.g. add that the individual is looking for a role in the job description's sector.
- Create hobbies that are relevant to the job description

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

# === Token Counting ===
def count_tokens(text: str, model_name: str = "gpt-4o-32k") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

# === LLM Call with Retry ===
@retry(
    wait=wait_random_exponential(min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError),
)
def call_agent(prompt):
    agent = get_conversational_agent(model_name="gpt-4o-32k")
    return agent.run(prompt)

# === Run Optimization ===
if st.button("🚀 Align CV"):
    token_count = count_tokens(st.session_state.prompt)
    st.info(f"📝 Prompt token count: **{token_count}**")

    max_tokens = 32768
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
                except json.JSONDecodeError:
                    st.warning("⚠️ The result isn't valid JSON. Showing raw output:")
                    st.code(result)
            except Exception as e:
                st.error(f"❌ Error from LLM: {e}")
