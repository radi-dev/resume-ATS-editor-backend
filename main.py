"""Main file"""

import asyncio
import logging
import sqlite3
import time
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

# Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# SQLite Setup
DB_FILE = "ai_resume_editor.db"


def init_db():
    """function to initialize the database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume TEXT,
            job_description TEXT,
            company_url TEXT,
            tailored_resume TEXT,
            cover_letter TEXT,
            email_template TEXT,
            interview_prep_points TEXT,
            scrape_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_created_at ON user_requests (created_at)"
    )
    conn.commit()
    conn.close()


init_db()


def save_to_db(data: dict):
    """function to save data to the database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_requests (
                resume, job_description, company_url,
                tailored_resume, cover_letter, email_template,
                interview_prep_points, scrape_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                data["resume"],
                data["job_description"],
                data["company_url"],
                data["tailored_resume"],
                data["cover_letter"],
                data["email_template"],
                data["interview_prep_points"],
                data["scrape_data"],
            ),
        )
        conn.commit()
    except sqlite3.Error as e:
        logging.error("Database error: %s", e)
    finally:
        conn.close()


# FastAPI Initialization
app = FastAPI()

# Model Configuration
# llm = OpenAI(temperature=0.7, model_name="gpt-4",api_key="sk-1ZQ7J9)


llm = ChatOpenAI(api_key=SecretStr("ugyuhk"), temperature=0.7, model="gpt-4o-mini")


# Input Data Models
class ResumeJobData(BaseModel):
    """function to save data to the database"""

    resume: str
    job_description: str
    company_url: str


# Prompt Templates
resume_prompt = PromptTemplate(
    input_variables=["resume", "job_description", "company_info"],
    template=(
        "You are an expert career advisor and ATS compliance specialist. Given the resume:\n"
        "{resume}\n\n"
        "and the job description:\n"
        "{job_description}\n\n"
        "along with the following company information:\n"
        "{company_info}\n\n"
        "Tailor the resume to match the job description and company information, ensuring ATS compliance."
        "Provide the updated resume in markdown format."
    ),
)

cover_letter_prompt = PromptTemplate(
    input_variables=["job_description", "company_info"],
    template=(
        "You are an expert career advisor. Write an ATS-compliant and tailored cover letter based on the following job description:\n"
        "{job_description}\n\n"
        "and the company information:\n"
        "{company_info}\n\n"
        "Provide the output in markdown format."
    ),
)

email_prompt = PromptTemplate(
    input_variables=["job_description", "company_info"],
    template=(
        "Write a professional email to accompany a job application. Use the following job description:\n"
        "{job_description}\n\n"
        "and the company information:\n"
        "{company_info}\n\n"
        "Provide the output in markdown format."
    ),
)

interview_prep_prompt = PromptTemplate(
    input_variables=["job_description", "company_info"],
    template=(
        "You are a career coach. Based on the job description:\n"
        "{job_description}\n\n"
        "and the company information:\n"
        "{company_info}\n\n"
        "Generate a list of potential interview questions and key points the candidate should prepare."
        "Provide the output in markdown format."
    ),
)


# Helper Functions
async def scrape_company_website(url: str) -> str:
    """function to scrape company website"""
    logging.info("Scraping company website: %s", url)

    async def fetch_page_content(page_url):
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(page_url)
                response.raise_for_status()
                logging.info("Successfully fetched content from %s", page_url)
                return response.text
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logging.error("Error fetching %s: %s", page_url, e)
                return ""

    async def extract_links(page_content, base_url):
        soup = BeautifulSoup(page_content, "html.parser")
        links = set()
        for anchor in soup.find_all("a", href=True):
            link = urljoin(base_url, anchor["href"])
            if base_url in link:
                links.add(link)
        return links

    visited = set()
    to_visit = {url}
    all_texts = []

    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue

        page_content = await fetch_page_content(current_url)
        if page_content:
            visited.add(current_url)
            all_texts.append(page_content)

            new_links = await extract_links(page_content, url)
            to_visit.update(new_links - visited)

        # Add delay to prevent server overload
        time.sleep(1)

    if not all_texts:
        logging.warning("No content scraped from %s", url)
        return ""

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text("\n".join(all_texts))
    combine_chain = StuffDocumentsChain(
        llm_chain=LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["text"],
                template="Summarize the following content:\n{text}\n",
            ),
        ),
        document_variable_name="text",
    )
    company_info = combine_chain.run(texts)
    logging.info("Scraping completed for %s", url)
    return company_info


async def generate_outputs(resume: str, job_description: str, company_info: str):
    """function to generate outputs"""
    logging.info(
        "Generating outputs from AI with resume length: %s characters...", len(resume)
    )
    resume_chain = LLMChain(llm=llm, prompt=resume_prompt)
    cover_letter_chain = LLMChain(llm=llm, prompt=cover_letter_prompt)
    email_chain = LLMChain(llm=llm, prompt=email_prompt)
    interview_prep_chain = LLMChain(llm=llm, prompt=interview_prep_prompt)

    tasks = [
        resume_chain.arun(
            resume=resume, job_description=job_description, company_info=company_info
        ),
        cover_letter_chain.arun(
            job_description=job_description, company_info=company_info
        ),
        email_chain.arun(job_description=job_description, company_info=company_info),
        interview_prep_chain.arun(
            job_description=job_description, company_info=company_info
        ),
    ]

    try:
        results = await asyncio.gather(*tasks)
        logging.info("AI outputs successfully generated.")
        return {
            "tailored_resume": results[0],
            "cover_letter": results[1],
            "email_template": results[2],
            "interview_prep_points": results[3],
        }
    except (httpx.HTTPStatusError, sqlite3.Error, asyncio.TimeoutError) as e:
        logging.error("Error generating outputs: %s", e)
        return {
            "tailored_resume": "",
            "cover_letter": "",
            "email_template": "",
            "interview_prep_points": "",
        }


# API Endpoint
@app.post("/process-resume")
async def process_resume(data: ResumeJobData):
    """Process resume and generate outputs."""
    logging.info(
        "Processing request with resume length: %s characters...", len(data.resume)
    )
    try:
        company_info = await scrape_company_website(data.company_url)
        outputs = await generate_outputs(
            resume=data.resume,
            job_description=data.job_description,
            company_info=company_info,
        )

        result_data = {
            "resume": data.resume,
            "job_description": data.job_description,
            "company_url": data.company_url,
            "tailored_resume": outputs["tailored_resume"],
            "cover_letter": outputs["cover_letter"],
            "email_template": outputs["email_template"],
            "interview_prep_points": outputs["interview_prep_points"],
            "scrape_data": company_info,
        }

        save_to_db(result_data)
        logging.info("Request processed and saved to database.")
        return outputs
    except httpx.HTTPStatusError as e:
        logging.error("HTTP error occurred: %s", e)
        return {"error": "Failed to fetch company information."}
    except sqlite3.Error as e:
        logging.error("Database error: %s", e)
        return {"error": "Failed to save data to the database."}
    except (
        httpx.RequestError,
        sqlite3.Error,
        asyncio.TimeoutError,
    ) as e:
        logging.error("Error occurred: %s", e)
        return {"error": "An error occurred while processing the request."}
