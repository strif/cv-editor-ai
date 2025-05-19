import streamlit as st
import time
from settings import PRESENTATION_ID
from gslides_utils import extract_slide_objects, apply_updates_to_slides
from llm_agent import get_conversational_agent

# Config
MAX_SLIDES = 5  # Limit how many slides to process
THROTTLE_SECONDS = 1  # Add delay between API calls to avoid rate limits

# Streamlit setup
st.set_page_config(page_title="AI CV Editor", layout="wide")
st.title("AI-Powered CV Editor (Google Slides + LangChain)")

# Inputs
job_url = st.text_input("🔗 Paste LinkedIn Job URL (optional)")
custom_prompt = st.text_area("🧠 Custom Prompt")
trigger = st.button("🧪 Generate Suggestions")

# Session state
if "preview" not in st.session_state:
    st.session_state.preview = None
if "objects" not in st.session_state:
    st.session_state.objects = None

if trigger:
    with st.spinner("📤 Extracting Slides Content..."):
        st.session_state.objects = extract_slide_objects(PRESENTATION_ID)

    total_found = len(st.session_state.objects)
    st.success(f"Found {total_found} text elements. Processing up to {MAX_SLIDES}.")

    agent = get_conversational_agent()
    preview = []

    with st.spinner("🤖 Generating Suggestions..."):
        progress = st.progress(0)
        for i, obj in enumerate(st.session_state.objects[:MAX_SLIDES]):
            text = obj['text']
            objectId = obj['objectId']
            full_prompt = f"""You are editing a CV. Here is a section of the slide:

---
{text}
---

Based on the job post: {job_url}
And this additional input: {custom_prompt}

Suggest an improved version of this text.
"""
            try:
                new_text = agent.run(full_prompt)
            except Exception as e:
                new_text = f"[Error generating text: {e}]"

            preview.append({
                "objectId": objectId,
                "old": text,
                "new": new_text
            })

            progress.progress((i + 1) / MAX_SLIDES)
            time.sleep(THROTTLE_SECONDS)

    st.session_state.preview = preview

# Show preview
if st.session_state.preview:
    st.markdown("### 📝 Preview Changes")
    for p in st.session_state.preview:
        st.markdown(f"""**Before:**  
{p['old']}

**After:**  
{p['new']}
---
""")

    if st.button("✅ Apply to Slides"):
        apply_updates_to_slides(
            PRESENTATION_ID,
            [{"objectId": p["objectId"], "new_text": p["new"]} for p in st.session_state.preview]
        )
        st.success("Slides updated successfully.")
