import streamlit as st
from settings import PRESENTATION_ID
from gslides_utils import extract_slide_objects, apply_updates_to_slides
from llm_agent import get_conversational_agent

st.set_page_config(page_title="AI CV Editor", layout="wide")
st.title("AI-Powered CV Editor (Google Slides + LangChain)")

# Input fields
job_url = st.text_input("🔗 Paste LinkedIn Job URL (optional)")
custom_prompt = st.text_area("🧠 Custom Prompt")
trigger = st.button("🧪 Generate Suggestions")

# Initialize session state
if "preview" not in st.session_state:
    st.session_state.preview = None
if "objects" not in st.session_state:
    st.session_state.objects = None

if trigger:
    with st.spinner("📤 Extracting Slides Content..."):
        st.session_state.objects = extract_slide_objects(PRESENTATION_ID)

    st.success(f"Found {len(st.session_state.objects)} text elements.")
    agent = get_conversational_agent()
    preview = []

    with st.spinner("🤖 Generating Suggestions..."):
        for obj in st.session_state.objects:
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
                preview.append({
                    "objectId": objectId,
                    "old": text,
                    "new": new_text
                })
            except Exception as e:
                preview.append({
                    "objectId": objectId,
                    "old": text,
                    "new": f"[Error generating text: {e}]"
                })

    st.session_state.preview = preview

# Preview Changes
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
