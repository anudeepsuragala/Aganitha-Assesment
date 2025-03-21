import streamlit as st
from langchain_community.document_loaders import WebBaseLoader


from chains import Chain
from portfolio import Portfolio
from utlis import clean_text


def create_streamlit_app(model, portfolio, clean_text):
    st.title("📧  Aganitha Cold Mail Generator")
    url_input = st.text_input("Enter a URL:", value="https://www.amazon.jobs/en/jobs/2916022/software-development-engineer-i")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = model.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = model.write_mail(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator by using llama ", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)