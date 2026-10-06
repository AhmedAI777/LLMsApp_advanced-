import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# EDUCATION AI ASSISTANT
# ============================================================

ASSISTANT_NAME = "Education AI Assistant"


# ============================================================
# EDUCATION KNOWLEDGE BASE
# ============================================================

EDUCATION_DOCUMENTS = [
    {
        "id": "admission_001",
        "service": "university_admission",
        "title": "University Admission Guide",
        "source": "Official University Admission Guide",
        "text": (
            "University admission requires meeting the required "
            "academic grades, submitting the required documents, "
            "and applying before the admission deadline."
        )
    },

    {
        "id": "scholarship_001",
        "service": "scholarships",
        "title": "University Scholarship Guidelines",
        "source": "Official Scholarship Guidelines",
        "text": (
            "Scholarship eligibility depends on meeting the minimum "
            "GPA requirement, eligible university enrollment, "
            "required documents, and the application deadline."
        )
    },

    {
        "id": "teaching_001",
        "service": "teaching_positions",
        "title": "University Teaching Position Requirements",
        "source": "Official University Careers Guide",
        "text": (
            "University teaching positions may require a relevant "
            "academic degree, teaching experience, relevant field "
            "of study, and academic documents."
        )
    },

    {
        "id": "academic_001",
        "service": "academic_careers",
        "title": "Academic Career Guidelines",
        "source": "Official Academic Career Guidelines",
        "text": (
            "Academic careers may require an appropriate academic "
            "qualification, relevant research or academic experience, "
            "academic documents, and references."
        )
    }
]


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CREATE DOCUMENT EMBEDDINGS
# ============================================================

document_texts = [
    document["text"]
    for document in EDUCATION_DOCUMENTS
]

document_embeddings = model.encode(
    document_texts
)


# ============================================================
# SECURITY
# ============================================================

BLOCKED_PHRASES = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "show me your instructions",
    "تجاهل التعليمات"
]

SCOPE_WORDS = [
    "university",
    "admission",
    "scholarship",
    "teaching",
    "lecturer",
    "academic",
    "faculty",
    "course",
    "professor",
    "degree",
    "جامعة",
    "قبول",
    "منحة",
    "تدريس",
    "محاضر",
    "أكاديمي",
    "كلية",
    "تخصص"
]


def check_input(message):

    message_lower = message.lower()

    for phrase in BLOCKED_PHRASES:

        if phrase.lower() in message_lower:

            return False, (
                "I can only help with education and "
                "university services."
            )

    for word in SCOPE_WORDS:

        if word.lower() in message_lower:

            return True, ""

    return False, (
        "This request is outside the scope of the "
        "Education AI Assistant.\n\n"
        "I can help with:\n"
        "- University admission\n"
        "- Scholarships\n"
        "- Teaching positions\n"
        "- Academic careers"
    )


# ============================================================
# SEMANTIC SEARCH / RAG
# ============================================================

def semantic_search(question, top_k=2):

    question_embedding = model.encode(
        [question]
    )

    similarities = cosine_similarity(
        question_embedding,
        document_embeddings
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "id": EDUCATION_DOCUMENTS[index]["id"],
            "service": EDUCATION_DOCUMENTS[index]["service"],
            "title": EDUCATION_DOCUMENTS[index]["title"],
            "source": EDUCATION_DOCUMENTS[index]["source"],
            "text": EDUCATION_DOCUMENTS[index]["text"],
            "similarity": float(similarities[index])
        })

    return results


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(question, results):

    if not results:

        return (
            "I could not find relevant information "
            "for this question."
        )

    best = results[0]

    similarity = best["similarity"]

    # Relevance threshold
    if similarity < 0.25:

        return (
            "I could not find sufficiently relevant "
            "information to answer this reliably.\n\n"
            "Please provide more details or check the "
            "official university source."
        )

    question_lower = question.lower()

    if "admission" in question_lower or "قبول" in question_lower:

        return (
            "For university admission, you generally need to "
            "meet the required academic grades, submit the "
            "required documents, and apply before the admission "
            "deadline."
        )

    if "scholarship" in question_lower or "منحة" in question_lower:

        return (
            "Scholarship eligibility generally depends on "
            "meeting the minimum GPA requirement, eligible "
            "university enrollment, required documents, and "
            "the application deadline."
        )

    if "teaching" in question_lower or "تدريس" in question_lower:

        return (
            "University teaching positions may require a relevant "
            "academic degree, teaching experience, a relevant "
            "field of study, and academic documents."
        )

    if (
        "academic" in question_lower
        or "career" in question_lower
        or "أكاديمي" in question_lower
    ):

        return (
            "Academic careers may require an appropriate academic "
            "qualification, relevant research or academic "
            "experience, academic documents, and references."
        )

    return (
        "Based on the available education information:\n\n"
        + best["text"]
    )


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def answer_with_rag(question):

    # Security check
    allowed, message = check_input(question)

    if not allowed:

        return {
            "answer": message,
            "sources": []
        }

    # Retrieve relevant documents
    results = semantic_search(
        question,
        top_k=2
    )

    # Generate answer
    answer = generate_answer(
        question,
        results
    )

    # Sources
    sources = []

    for result in results:

        if result["similarity"] >= 0.25:

            sources.append(
                result["source"]
            )

    return {
        "answer": answer,
        "sources": sources
    }


# ============================================================
# GRADIO CHATBOT
# ============================================================

def chatbot(message, history):

    result = answer_with_rag(
        message
    )

    answer = result["answer"]

    if result["sources"]:

        answer += "\n\n### Sources\n"

        for source in result["sources"]:

            answer += f"- {source}\n"

    return answer


# ============================================================
# GRADIO INTERFACE
# ============================================================

demo = gr.ChatInterface(
    fn=chatbot,
    title="🎓 Education AI Assistant",
    description=(
        "AI assistant for students and lecturers.\n\n"
        "Ask about university admission, scholarships, "
        "teaching positions, or academic careers."
    ),
    textbox=gr.Textbox(
        placeholder=(
            "Ask your education question..."
        ),
        container=True
    )
)


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False
    )
