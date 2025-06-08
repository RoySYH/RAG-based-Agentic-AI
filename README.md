# RAG-based Agentic AI for Meeting Room Booking

## Overview
This project implements an Agentic AI that answers questions about meeting room booking policies using Retrieval-Augmented Generation (RAG) and makes decisions on time slot availability. It combines rule-based logic for time conflict checks with generative AI (Gemini API) for policy-related queries, using English prompts for stable responses. API keys are securely managed using `python-dotenv`.

## Features
- **Policy Queries**: Retrieves information from a knowledge base (`policy.txt`) to answer questions like "How to book a meeting room?".
- **Time Slot Decisions**: Checks time slot availability (e.g., "Is 10:00 AM available?") and suggests alternatives for conflicts.
- **English Interface**: Uses English prompts and responses for robust generation.
- **RAG Pipeline**: Integrates FAISS for vector search and Gemini API for answer generation.
- **Secure API Management**: Uses `python-dotenv` to store sensitive API keys.

## Tech Stack
- **Python**: Core programming language.
- **LangChain Community**: For document loading, text splitting, and vector search.
- **LangChain Google GenAI**: Integrates Gemini API for generative responses.
- **Google Generative AI**: Provides Gemini API access.
- **FAISS**: Efficient vector search for RAG.
- **Sentence Transformers**: Generates multilingual embeddings (`all-MiniLM-L6-v2`).
- **python-dotenv**: Manages API keys securely.

## Setup
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd rag-agentic-ai
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install langchain-community langchain-huggingface langchain-google-genai google-generativeai faiss-cpu sentence-transformers python-dotenv
   ```
4. Create a `.env` file in the project root with your Gemini API key:
   ```
   GOOGLE_API_KEY=your-gemini-api-key
   ```
5. Run the script:
   ```bash
   python rag_agent.py
   ```

## Usage
- Input questions like:
  - "How to book a meeting room?"
  - "Can I book meeting room A at 10:00 AM tomorrow?"
  - "Is 11:00 AM available?"
- Output is saved to `output.txt`.

## Demo Output
```
Q: How to book a meeting room?
A: All employees must book meeting rooms through the system, selecting the date and time at least 24 hours in advance. Maximum booking time is 2 hours.
Q: Can I book meeting room A at 10:00 AM tomorrow?
A: Time slot 10:00 AM is booked, Suggested alternative times: 11:00 AM or 3:00 PM
Q: Is 11:00 AM available?
A: Time slot 11:00 AM is available
```

## Lessons Learned
- Implemented RAG using LangChain and FAISS for efficient knowledge retrieval.
- Designed an Agentic AI combining rule-based logic and generative AI (Gemini API).
- Used English prompts to overcome language barriers, ensuring stable responses.
- Securely managed API keys with `python-dotenv`.
- Debugged issues like tokenizer errors, model compatibility, and encoding problems.
- Managed complex dependencies in a Windows environment.

## License
MIT