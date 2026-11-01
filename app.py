import requests
import streamlit as st

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Assistant Factures", page_icon="🧾")
st.title("🧾 Assistant Factures")

question = st.text_input("Votre question", placeholder="Ex : Quel est le montant total de la facture F-2024-001 ?")

if st.button("Envoyer", type="primary") and question.strip():
    with st.spinner("Recherche en cours..."):
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=120)
            response.raise_for_status()
            st.markdown(response.json().get("answer", ""))
        except requests.exceptions.ConnectionError:
            st.error("API injoignable : lancez d'abord `python main.py`.")
        except Exception as e:
            st.error(f"Erreur : {e}")