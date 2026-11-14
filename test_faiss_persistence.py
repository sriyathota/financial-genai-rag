import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from unittest.mock import patch, sentinel

from LocalRAG import (
    get_vectorstore,
    append_documents_to_vectorstore,
    create_whole_pipeline,
)


class TinyEmbeddings(Embeddings):
    def __init__(self):
        self.batches = []

    def embed_documents(self, texts):
        texts = list(texts)
        self.batches.append(texts)
        return [
            [float(len(text)), float(sum(map(ord, text)) % 1000)]
            for text in texts
        ]

    def embed_query(self, text):
        return [float(len(text)), float(sum(map(ord, text)) % 1000)]

embedder = TinyEmbeddings()

with tempfile.TemporaryDirectory() as temp_dir:
    index_dir = Path(temp_dir)

    built = get_vectorstore(
        chunked_data=["An existing trading passage."],
        embedder=embedder,
        save_path=index_dir,
    )
    assert built.index.ntotal == 1

    loaded = get_vectorstore(load_path=index_dir, embedder=embedder)
    assert loaded.index.ntotal == 1

    appended = append_documents_to_vectorstore(
        new_documents=[
            Document(
                page_content="A newly added trading passage.",
                metadata={"source": "new_book.pdf", "page": 3},
            )
        ],
        embedder=embedder,
        index_dir=index_dir,
    )
    assert appended.index.ntotal == 2

    reloaded = get_vectorstore(load_path=index_dir, embedder=embedder)
    assert reloaded.index.ntotal == 2

    new_doc_id = reloaded.index_to_docstore_id[1]
    new_doc = reloaded.docstore.search(new_doc_id)
    assert new_doc.metadata == {"source": "new_book.pdf", "page": 3}

    assert embedder.batches == [
        ["An existing trading passage."],
        ["A newly added trading passage."],
    ]
    with (
        patch("LocalRAG.get_embeddings", return_value=embedder),
        patch(
            "LocalRAG.read_trading_books",
            side_effect=AssertionError("PDFs were read again"),
        ),
        patch("LocalRAG.get_retreiver", return_value=object()),
        patch("LocalRAG.create_rag_chain", return_value=sentinel.chain),
    ):
        chain = create_whole_pipeline(
            system_prompt="Test prompt",
            documents_dir=index_dir / "unused_books",
            vectorstore_save_path=index_dir,
        )

    assert chain is sentinel.chain
    assert embedder.batches == [
        ["An existing trading passage."],
        ["A newly added trading passage."],
    ]

print("PASS: build → reload → append → reload; metadata preserved")
