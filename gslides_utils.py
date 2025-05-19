from google.oauth2 import service_account
from googleapiclient.discovery import build
import streamlit as st

def get_slides_service():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/presentations"]
    )
    return build("slides", "v1", credentials=creds)

def extract_slide_objects(presentation_id):
    service = get_slides_service()
    pres = service.presentations().get(presentationId=presentation_id).execute()
    objects = []

    for slide in pres.get("slides", []):
        for el in slide.get("pageElements", []):
            shape = el.get("shape")
            if shape and shape.get("text"):
                text = "".join(
                    r.get("textRun", {}).get("content", "") for r in shape["text"].get("textElements", [])
                ).strip()
                if text:
                    objects.append({ "objectId": el["objectId"], "text": text })

    return objects

def apply_updates_to_slides(presentation_id, updates):
    service = get_slides_service()
    requests = []

    for update in updates:
        requests.extend([
            { "deleteText": { "objectId": update["objectId"], "textRange": {"type": "ALL"} } },
            { "insertText": { "objectId": update["objectId"], "insertionIndex": 0, "text": update["new_text"] } }
        ])

    service.presentations().batchUpdate(
        presentationId=presentation_id, body={"requests": requests}
    ).execute()
