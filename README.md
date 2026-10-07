# 🎓 Education AI Assistant

An intelligent, evidence-based AI assistant designed to support **university students, lecturers, and academic professionals** through university admission, scholarship, teaching-position, and academic-career guidance.

The system combines **Retrieval-Augmented Generation (RAG)**, semantic search, document evidence, application management, security controls, and human review into a single education-focused platform.

---

## 📌 Project Overview

Students and academic professionals often need to search through different documents and requirements when applying for university admission, scholarships, teaching positions, or academic career opportunities.

Traditional approaches require users to:

- Search through multiple documents manually.
- Understand complicated eligibility requirements.
- Identify which documents are required.
- Determine whether an application is complete.
- Track application status separately.
- Contact an advisor for basic questions.
- Risk receiving inaccurate information from a generic AI chatbot.

The **Education AI Assistant** addresses these problems by providing a centralized intelligent system that combines educational knowledge retrieval with a controlled application workflow.

---

# 🎯 Problem

The main problem is the lack of a unified intelligent platform that can provide **reliable educational guidance while maintaining evidence, security, and human control over application decisions**.

A general-purpose chatbot may generate an answer that sounds correct but is not supported by an official or approved source.

This creates several risks:

### 1. Hallucinated information

An AI model may invent:

- Admission requirements
- Scholarship requirements
- Application deadlines
- Academic qualifications
- Teaching requirements
- Career requirements

### 2. Information scattered across documents

Students and lecturers may need to search through multiple documents to determine:

- Eligibility
- Required qualifications
- Required documents
- Application procedures
- Deadlines
- Review rules

### 3. No application workflow

A normal chatbot cannot reliably manage:

- Student applications
- Lecturer applications
- Supporting documents
- Application status
- Reviewer decisions
- Rejection reasons

### 4. Privacy and security risks

Education applications may contain sensitive information such as:

- Name
- Email
- Phone number
- Identification information
- Academic records
- GPA
- Degree information
- Research experience
- Teaching experience
- References
- Supporting documents

### 5. Lack of human oversight

AI should provide guidance, but it should not independently make important admission, scholarship, hiring, or academic-career decisions.

6. The Architecture
User Interface
- User
- Gradio Web UI
Main System Paths
- AI Assistant
- Applications
AI Assistant
1. Security Layer – checks and protects user requests.
2. Service Router – identifies the correct education service.
3. RAG Pipeline – retrieves relevant information.
4. PDF Files – provide the knowledge base.
5. Embedding Model – converts text into vectors.
6. Chroma Vector Database – stores and searches the vectors.
7. Semantic Retrieval – finds the most relevant information.
8. Relevance Check – checks whether the information is useful.
9. Conflict Check – detects conflicting information.
10. LLM – generates the answer.
11. Evidence + Confidence – validates the response.
12. Final Answer – presented to the user with sources.
Application System
1. Student / Educator submits application
2. SQLite Database – stores application data.
3. Reviewer Panel – allows authorized human review.
4. Final Decision – Approved or Rejected.

   “My architecture has two main paths: an AI Assistant and an Application System. The AI Assistant uses security, service routing, and RAG to retrieve relevant information from PDFs before generating an evidence-based answer. The Application System uses SQLite and a human reviewer to manage applications and make the final decision.”

7. The future work will let my bot decide rather than interaction advisor. Therefore, the chatbot will be Agent AI!!! can make decision and automatically proceed with the apllication and services with Students and Lecturers.

---

