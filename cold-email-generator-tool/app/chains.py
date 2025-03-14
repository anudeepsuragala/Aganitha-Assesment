import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.model = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.model
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
    
            ### INSTRUCTIONS:
            you are Anudeep , a business development at Aganitha, Aganitha is an ai & software consulting company dedicated seamless intergration of business process  through automated tools.
            over our experience, we have empowered numerous enterprices   with tailored solution process , optimization , cost reduction , increase efficency.
            Your job is to write cold mail to the client  regarding the job mentioned above describing the capability  in their fulfilling their needs
            also add the most relevent ones from the following links  to show case Aganitha portfolio:{link_list} 
            remeber you are anudeep , bde at Aganitha
            Do not provide preamble.
            ### EMAIL (NO PREAMBLE):
    
            """
        )
        chain_email = prompt_email | self.model
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))