import os
import html
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

CHROMA_PATH = "chroma_db"
embedding_function = OpenAIEmbeddings()

def _load_or_build_chroma():
    if os.path.exists(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        if db._collection.count() > 0:
            return db

    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    return Chroma.from_documents(documents, embedding_function, persist_directory=CHROMA_PATH)

db_books = _load_or_build_chroma()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 6,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


SYSTEM_PROMPT = (
    "You are a helpful book advisor. The user is browsing book recommendations "
    "shown in a gallery beside this chat. Your job is to help them refine their "
    "search through conversation.\n\n"
    "IMPORTANT: You may ONLY mention books that appear in the provided retrieved "
    "book data. NEVER recommend or reference books from your own knowledge that "
    "are not in the retrieved list.\n\n"
    "On the FIRST message: briefly acknowledge the type of books found, highlight "
    "one or two standouts from the retrieved list, then ask a follow-up question "
    "to help narrow down their preferences (e.g., sub-genre, mood, themes, "
    "time period).\n\n"
    "On FOLLOW-UP messages: use the user's refinement to comment on the updated "
    "results and ask another refining question if appropriate. Keep responses "
    "concise (2-4 sentences). Do NOT list every book -- the user can see them "
    "in the gallery."
)

EMOTION_COLS = ["joy", "surprise", "anger", "fear", "sadness"]


def _format_author(authors_raw: str) -> str:
    authors_split = authors_raw.split(";")
    if len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    elif len(authors_split) > 2:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    return authors_raw


def _build_book_context(recommendations: pd.DataFrame) -> str:
    context_parts = []
    for _, row in recommendations.iterrows():
        dominant_emotion = max(EMOTION_COLS, key=lambda e: row.get(e, 0))
        context_parts.append(
            f"Title: {row['title']}\n"
            f"Author: {_format_author(row['authors'])}\n"
            f"Category: {row.get('simple_categories', 'N/A')}\n"
            f"Dominant emotion: {dominant_emotion}\n"
            f"Description: {row['description']}\n"
        )
    return "\n---\n".join(context_parts)


def _build_book_cards_html(recommendations: pd.DataFrame) -> str:
    count = len(recommendations)
    cards_html = ""
    for _, row in recommendations.iterrows():
        title = html.escape(str(row["title"]))
        author = html.escape(_format_author(str(row["authors"])))
        rating = row.get("average_rating", 0)
        thumbnail = row["large_thumbnail"]

        cards_html += f"""
        <div class="book-card">
          <div class="book-cover">
            <img src="{thumbnail}" alt="{title}" loading="lazy"
                 onerror="this.style.display='none'" />
            <div class="cover-overlay">
              <div class="cover-title">{title}</div>
              <div class="cover-author">{author}</div>
            </div>
          </div>
          <div class="book-meta">
            <div class="meta-title">{title}</div>
            <div class="meta-author">{author}</div>
            <div class="meta-rating">&#9733; {rating}</div>
          </div>
        </div>
        """

    return f"""
    <div class="books-header">
      <h3>Recommended Books</h3>
      <span class="books-count">{count} titles</span>
    </div>
    <div class="books-grid">{cards_html}</div>
    """


EMPTY_BOOKS_HTML = """
<div class="books-header">
  <h3>Recommended Books</h3>
</div>
<div class="books-empty">
  <p>Enter a description above to discover books</p>
</div>
"""


def _ask_llm(chat_history: list, book_context: str, user_query: str, category: str, tone: str) -> str:
    context_message = (
        f"[Retrieved books for the query below]\n"
        f"Filters — Category: {category}, Tone: {tone}\n\n"
        f"{book_context}"
    )

    messages = [("system", SYSTEM_PROMPT)]
    for msg in chat_history:
        role = "human" if msg["role"] == "user" else msg["role"]
        messages.append((role, msg["content"]))
    messages.append(("human", f"{user_query}\n\n{context_message}"))

    return llm.invoke(messages).content


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    book_cards_html = _build_book_cards_html(recommendations)
    book_context = _build_book_context(recommendations)

    response_text = _ask_llm([], book_context, query, category, tone)

    chat_history = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response_text},
    ]

    return chat_history, book_cards_html, ""


def refine_recommendations(
        message: str,
        chat_history: list,
        category: str,
        tone: str,
):
    if not message or not message.strip():
        return chat_history, gr.update(), ""

    recommendations = retrieve_semantic_recommendations(message, category, tone)
    book_cards_html = _build_book_cards_html(recommendations)
    book_context = _build_book_context(recommendations)

    response_text = _ask_llm(chat_history, book_context, message, category, tone)

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_text},
    ]

    return chat_history, book_cards_html, ""


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ---- Global ---- */
.gradio-container {
    background-color: #F5F0E8 !important;
    font-family: 'Inter', sans-serif !important;
    max-width: 1400px !important;
}

/* ---- Header ---- */
.shelf-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 24px 0 20px 0;
}
.shelf-logo {
    width: 48px; height: 48px;
    background: #2C1810;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
}
.shelf-logo-inner {
    width: 20px; height: 20px;
    background: #C4956A;
    border-radius: 4px;
}
.shelf-brand h1 {
    margin: 0; font-size: 28px; font-weight: 800;
    color: #2C1810; letter-spacing: -0.5px;
}
.shelf-brand p {
    margin: 2px 0 0 0; font-size: 11px; font-weight: 600;
    color: #8B7355; letter-spacing: 0.15em; text-transform: uppercase;
}

/* ---- Input bar ---- */
#search-bar {
    background: #FFFDF9; border: 1px solid #E0D5C5;
    border-radius: 14px; padding: 20px 24px;
}
#search-bar .gr-input, #search-bar input, #search-bar textarea {
    background: #2C1810 !important; color: #F5F0E8 !important;
    border: none !important; border-radius: 8px !important;
}
#search-bar label, #search-bar .label-wrap span {
    text-transform: uppercase !important; font-size: 11px !important;
    font-weight: 700 !important; letter-spacing: 0.1em !important;
    color: #6B5D4F !important;
}
#search-bar select, #search-bar .wrap-inner {
    background: #2C1810 !important; color: #F5F0E8 !important;
    border: none !important; border-radius: 8px !important;
}
#search-bar .secondary {
    background: #2C1810 !important; color: #F5F0E8 !important;
}
#submit-btn {
    background: #5C6B4A !important; color: #FFFDF9 !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 14px !important;
    letter-spacing: 0.02em; padding: 12px 28px !important;
    min-height: 46px !important;
    transition: background 0.2s ease;
}
#submit-btn:hover {
    background: #4A5A3A !important;
}

/* ---- Chat panel ---- */
.advisor-header {
    display: flex; align-items: center; gap: 12px;
    padding: 16px 0 8px 0;
}
.advisor-avatar {
    width: 40px; height: 40px; background: #2C1810;
    border-radius: 10px; display: flex;
    align-items: center; justify-content: center;
}
.advisor-avatar-inner {
    width: 16px; height: 16px; background: #C4956A; border-radius: 3px;
}
.advisor-info { flex: 1; }
.advisor-info h4 {
    margin: 0; font-size: 15px; font-weight: 700; color: #2C1810;
}
.advisor-info p {
    margin: 1px 0 0 0; font-size: 11px; color: #8B7355;
}
.advisor-status {
    width: 10px; height: 10px; background: #6B9E4A;
    border-radius: 50%; margin-left: auto;
}

#chatbot {
    border: 1px solid #E0D5C5 !important;
    border-radius: 14px !important;
    background: #2C1810 !important;
}
#chatbot .message, #chatbot .message p,
#chatbot .message span, #chatbot .message li {
    font-size: 14px !important;
}
#chatbot .bot, #chatbot .assistant, #chatbot [data-testid="bot"] {
    background: #F0EBE1 !important;
    color: #2C1810 !important;
    border-radius: 12px !important;
}
#chatbot .bot p, #chatbot .assistant p, #chatbot [data-testid="bot"] p,
#chatbot .bot span, #chatbot .assistant span, #chatbot [data-testid="bot"] span,
#chatbot .bot li, #chatbot .assistant li, #chatbot [data-testid="bot"] li {
    color: #2C1810 !important;
}
#chatbot .user, #chatbot [data-testid="user"] {
    background: #F5F0E8 !important;
    color: #2C1810 !important;
    border-radius: 12px !important;
}
#chatbot .user p, #chatbot [data-testid="user"] p,
#chatbot .user span, #chatbot [data-testid="user"] span {
    color: #2C1810 !important;
}

#chat-input textarea {
    background: #FFFDF9 !important; border: 1px solid #E0D5C5 !important;
    border-radius: 24px !important; font-size: 13px !important;
    color: #2C1810 !important;
    padding: 12px 20px !important;
}
#chat-input textarea::placeholder {
    color: #A89880 !important; font-size: 13px !important;
}
#send-btn {
    background: #2C1810 !important; color: #F5F0E8 !important;
    border: none !important; border-radius: 50% !important;
    min-width: 44px !important; max-width: 44px !important;
    min-height: 44px !important; max-height: 44px !important;
    padding: 0 !important; font-size: 18px !important;
    display: flex !important; align-items: center !important;
    justify-content: center !important;
    transition: background 0.2s ease;
}
#send-btn:hover {
    background: #4A3828 !important;
}

/* ---- Book grid ---- */
.books-header {
    display: flex; align-items: center; gap: 12px;
    padding: 16px 0 12px 0;
}
.books-header h3 {
    margin: 0; font-size: 20px; font-weight: 700; color: #2C1810;
}
.books-count {
    background: #E8F5E0; color: #4A7A2A; font-size: 12px;
    font-weight: 700; padding: 4px 12px; border-radius: 20px;
}
.books-empty {
    display: flex; align-items: center; justify-content: center;
    min-height: 300px; color: #8B7355; font-size: 15px;
    background: #FFFDF9; border: 1px solid #E0D5C5; border-radius: 14px;
}
.books-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    max-height: 700px;
    overflow-y: auto;
    padding-right: 8px;
}
.books-grid::-webkit-scrollbar { width: 6px; }
.books-grid::-webkit-scrollbar-thumb {
    background: #C4B8A8; border-radius: 3px;
}
.book-card {
    background: #FFFDF9; border: 1px solid #E0D5C5;
    border-radius: 14px; overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.book-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(44, 24, 16, 0.12);
}
.book-cover {
    position: relative; width: 100%; aspect-ratio: 3/4;
    overflow: hidden; background: #3D2E22;
}
.book-cover img {
    width: 100%; height: 100%; object-fit: cover;
}
.cover-overlay {
    position: absolute; bottom: 0; left: 0; right: 0;
    padding: 40px 14px 14px 14px;
    background: linear-gradient(transparent, rgba(44, 24, 16, 0.85));
    color: #FFFDF9;
}
.cover-title {
    font-size: 14px; font-weight: 700; line-height: 1.3;
    margin-bottom: 2px;
}
.cover-author {
    font-size: 12px; opacity: 0.85;
}
.book-meta {
    padding: 12px 14px 14px 14px;
}
.meta-title {
    font-size: 13px; font-weight: 700; color: #2C1810;
    line-height: 1.3; margin-bottom: 2px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.meta-author {
    font-size: 12px; color: #8B7355; margin-bottom: 6px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.meta-rating {
    font-size: 12px; color: #C4956A; font-weight: 700;
}
"""

# ---------------------------------------------------------------------------
# Gradio theme
# ---------------------------------------------------------------------------
shelf_theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#faf6f0", c100="#f5f0e8", c200="#e8dfd2",
        c300="#d4c8b6", c400="#b8a890", c500="#8b7355",
        c600="#6b5d4f", c700="#4a3828", c800="#2c1810",
        c900="#1a0e08", c950="#0d0704",
    ),
    secondary_hue=gr.themes.Color(
        c50="#f0f5eb", c100="#e8f5e0", c200="#d0e6c0",
        c300="#a8cc8a", c400="#7eb35c", c500="#5c6b4a",
        c600="#4a5a3a", c700="#3a4a2c", c800="#2a3a1e",
        c900="#1a2a10", c950="#0d1508",
    ),
    neutral_hue=gr.themes.Color(
        c50="#faf6f0", c100="#f5f0e8", c200="#e0d5c5",
        c300="#c4b8a8", c400="#a89888", c500="#8b7355",
        c600="#6b5d4f", c700="#4a3828", c800="#2c1810",
        c900="#1a0e08", c950="#0d0704",
    ),
    font=gr.themes.GoogleFont("Inter"),
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
HEADER_HTML = """
<div class="shelf-header">
  <div class="shelf-logo"><div class="shelf-logo-inner"></div></div>
  <div class="shelf-brand">
    <h1>Shelf</h1>
    <p>Semantic Book Discovery</p>
  </div>
</div>
"""

ADVISOR_HEADER_HTML = """
<div class="advisor-header">
  <div class="advisor-avatar"><div class="advisor-avatar-inner"></div></div>
  <div class="advisor-info">
    <h4>Book Advisor</h4>
  </div>
  <div class="advisor-status"></div>
</div>
"""

with gr.Blocks(css=CUSTOM_CSS) as dashboard:
    gr.HTML(HEADER_HTML)

    with gr.Row(elem_id="search-bar"):
        user_query = gr.Textbox(
            label="Describe a Book",
            placeholder="e.g., A story about forgiveness",
            scale=3,
        )
        category_dropdown = gr.Dropdown(
            choices=categories, label="Category", value="All", scale=1,
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, label="Emotional Tone", value="All", scale=1,
        )
        submit_button = gr.Button(
            "Find Recommendations \u2192", elem_id="submit-btn", scale=1,
        )

    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML(ADVISOR_HEADER_HTML)
            chatbot = gr.Chatbot(elem_id="chatbot", height=480, show_label=False)
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Refine your search...",
                    show_label=False,
                    scale=6,
                    elem_id="chat-input",
                )
                send_button = gr.Button("\u27A4", elem_id="send-btn", scale=0)

        with gr.Column(scale=3):
            book_output = gr.HTML(value=EMPTY_BOOKS_HTML)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[chatbot, book_output, chat_input],
    )

    send_button.click(
        fn=refine_recommendations,
        inputs=[chat_input, chatbot, category_dropdown, tone_dropdown],
        outputs=[chatbot, book_output, chat_input],
    )

    chat_input.submit(
        fn=refine_recommendations,
        inputs=[chat_input, chatbot, category_dropdown, tone_dropdown],
        outputs=[chatbot, book_output, chat_input],
    )


if __name__ == "__main__":
    dashboard.launch(theme=shelf_theme)
