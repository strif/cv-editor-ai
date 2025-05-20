import streamlit as st
import json
import time
from llm_agent import get_conversational_agent
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai._exceptions import RateLimitError
import tiktoken

st.set_page_config(page_title="AI CV JSON Optimizer", layout="wide")
st.title("📄 AI-Powered JSON CV Optimizer")

# CV data (hardcoded for now)
cv_data = {
  "personal_info": {
    "name": "Kostas Voudouris",
    "headline": "Product | Marketing | Innovation",
    "contact": {
      "address": "58 Elvendon Road, London, United Kingdom",
      "phone": "07530525525",
      "email": "kostantinosv@gmail.com"
    },
    "nationality": "British / Greek",
    "linkedin_profile": "",
    "profile_description": "Dedicated to driving product innovation and improving performance metrics through extensive leadership experience. Passionate about using technology to deliver impactful solutions that fuel growth in fast-paced environments. Eager to contribute to a visionary team shaping the future of digital marketing and product development.",
    "hobbies": "",
    "skills": [
      "Team Management",
      "Early Stage Innovation & AI",
      "Product Management",
      "Python",
      "Google Cloud",
      "AWS",
      "Ad Platforms"
    ]
  },
  "employment_history": [
    {
      "company": "Choreograph",
      "title": "SVP, Product (Commerce)",
      "from": "July 2024",
      "to": "Present",
      "location": "London",
      "responsibilities_or_achievements": [
        "Leads a 30-person multidisciplinary team spanning product, engineering, data science, UX, and analytics.",
        "Drives the global vision and strategy for commerce products, aligning cross-functional stakeholders and resources around a unified portfolio.",
        "Oversees budgeting, prioritization, and execution of high-impact initiatives across the commerce ecosystem."
      ]
    },
    {
      "company": "EssenceMediacom",
      "title": "VP of Product Innovation",
      "from": "November 2022",
      "to": "May 2024",
      "location": "",
      "responsibilities_or_achievements": [
        "Executed innovation frameworks, translating vision into high-impact products.",
        "Prioritized business challenges, driving transformative AI-driven solutions. Led GenAI cohorts, offering training on LLMs and AI best practices.",
        "Formed and directed cross-functional teams, delivering iterative Proof of Concepts, MVPs, and successful product launches."
      ]
    },
    {
      "company": "Essence",
      "title": "Business Owner - OCTRA (Product)",
      "from": "May 2020",
      "to": "June 2024",
      "location": "",
      "responsibilities_or_achievements": [
        "Pioneered OCTRA, a GroupM-level USP used by 80+ global clients.",
        "Led a 45-person team across product, engineering, design, QA, and customer success.",
        "Oversaw pricing strategy, forecasting, and revenue reporting, while establishing feedback loops to keep the roadmap aligned with client needs."
      ]
    },
    {
      "company": "Essence Global",
      "title": "Global Head of Media Technology",
      "from": "January 2018",
      "to": "June 2024",
      "location": "London",
      "responsibilities_or_achievements": [
        "Led a global team of 15, building advanced software to tackle complex media challenges.",
        "Acted as a key technology stakeholder, shaping strategic decisions and the tech vision.",
        "Mentored internal talent, providing hands-on training in agile development."
      ]
    },
    {
      "company": "Maxus - Essence Global",
      "title": "Head of Organic & Content Performance",
      "from": "November 2011",
      "to": "December 2017",
      "location": "London",
      "responsibilities_or_achievements": [
        "Responsible for the P&L of a 20-member team introducing new services at global scale.",
        "Services included SEO, Conversion Rate Optimization, and Performance Content.",
        "Led recruiting, pitching, onboarding, and servicing clients including Apple, Burberry & UPS."
      ]
    }
  ],
  "education": [
    {
      "degree": "Computer Science and AI",
      "university": "City, University of London",
      "from": "January 2008",
      "to": "January 2011",
      "notes": "Graduated with a First Class Degree"
    }
  ]
}

# Input: Optional job description
st.subheader("Optional Job Role to Tailor For")

# Load job_desc from session state or default empty string
job_desc = st.text_area("Paste the job description or target role (optional):", value=st.session_state.get("job_desc", ""))

# Update session state and prompt dynamically when job_desc changes
if job_desc != st.session_state.get("job_desc", ""):
    st.session_state.job_desc = job_desc
    st.session_state.prompt = None  # Reset prompt so it rebuilds below

# Function to create the prompt string
def create_prompt(cv_json, job_description):
    return f"""
You are an expert career advisor helping improve a JSON-based CV.

Below is a user's CV in JSON format. Rewrite and enhance this CV:
- Highlight key skills and achievements for leadership/product roles.
- Tailor it towards modern tech/innovation/product environments.
- Keep it in valid JSON structure.
- Prioritize clarity, brevity, and impact.
- Do not omit important responsibilities unless redundant.
- If a job description is provided, align it accordingly.

{"Here is the job description: " + job_description if job_description else ""}

CV JSON:
{json.dumps(cv_json, indent=2)}
"""

# Build prompt if not already in session state
if "prompt" not in st.session_state or st.session_state.prompt is None:
    st.session_state.prompt = create_prompt(cv_data, st.session_state.get("job_desc", ""))

st.subheader("Edit the prompt to customize your CV optimization")
prompt = st.text_area("Prompt:", value=st.session_state.prompt, height=400)

# Update session state if user edits the prompt manually
if prompt != st.session_state.prompt:
    st.session_state.prompt = prompt

# Function to count tokens for the prompt
def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

# New function to get tokens decoded as strings for display
def get_tokens(text: str, model_name: str = "gpt-4o-mini") -> list[str]:
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    token_strings = [encoding.decode([t]) for t in tokens]
    return token_strings

# Retry logic for API calls
@retry(
    wait=wait_random_exponential(min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError),
)
def call_agent(prompt):
    agent = get_conversational_agent()
    return agent.run(prompt)

if st.button("🚀 Optimize CV JSON"):
    token_count = count_tokens(st.session_state.prompt)
    tokens = get_tokens(st.session_state.prompt)
    st.info(f"📝 Prompt token count: **{token_count}**")

    # Show actual tokens in an expandable section
    with st.expander("Show tokens"):
        st.write(tokens)

    max_tokens = 16385
    if token_count > max_tokens:
        st.error(f"❌ Your prompt is too long by {token_count - max_tokens} tokens. Please shorten the CV or job description or prompt.")
    else:
        with st.spinner("Calling LLM to optimize your CV JSON..."):
            try:
                result = call_agent(st.session_state.prompt)

                # Try to parse as JSON
                try:
                    parsed = json.loads(result)
                    st.success("✅ Valid JSON returned!")
                    st.code(json.dumps(parsed, indent=2), language="json")
                except json.JSONDecodeError:
                    st.warning("⚠️ The result isn't valid JSON. Showing raw output:")
                    st.code(result)

            except Exception as e:
                st.error(f"❌ Error from LLM: {e}")
