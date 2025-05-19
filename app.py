import streamlit as st
from settings import PRESENTATION_ID
from gslides_utils import extract_slide_objects, apply_updates_to_slides
from llm_agent import get_conversational_agent

st.set_page_config(page_title="AI CV Editor", layout="wide")
st.title("AI-Powered CV Editor (Google Slides + LangChain)")

job_url = st.text_input("🔗 Paste LinkedIn Job URL (optional)")
custom_prompt = st.text_area("🧠 Custom Prompt")
trigger = st.button("🧪 Generate Suggestions")

if trigger:
    st.markdown("### 📤 Extracting Slides Content...")
    objects = extract_slide_objects(PRESENTATION_ID)
    st.write(f"Found {len(objects)} text elements.")

    agent = get_conversational_agent()
    preview = []

    for obj in objects:
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
        new_text = agent.run(full_prompt)
        preview.append({ "objectId": objectId, "old": text, "new": new_text })

    st.markdown("### 📝 Preview Changes")
    for p in preview:
        st.markdown(f"""**Before:**  
{p['old']}

**After:**  
{p['new']}
---
""")

    if st.button("✅ Apply to Slides"):
        apply_updates_to_slides(
            PRESENTATION_ID,
            [{"objectId": p["objectId"], "new_text": p["new"]} for p in preview]
        )
        st.success("Slides updated successfully.")
