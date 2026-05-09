# src/ingestion/chunker.py

import re
from typing import List, Dict, Tuple
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter


class PolicyChunker:
    """
    Advanced Markdown-aware semantic chunker for corporate policy documents.
    Handles .txt files with # and ## headers, preserves tables, and adds overlap.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap_percent: float = 0.12,  # 12% overlap (within 10-15% range)
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = int(chunk_size * chunk_overlap_percent)
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+?)$', re.MULTILINE)

    def _extract_headers_and_content(self, text: str) -> List[Dict]:
        """
        Splits document by headers (#, ##, ### etc.) while preserving hierarchy.
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = {"header": "", "h1": "", "h2": "", "content": []}
        current_h1 = ""
        current_h2 = ""

        for line in lines:
            header_match = self.header_pattern.match(line.strip())

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Save previous chunk if it has content
                if current_chunk["content"] and current_chunk["header"]:
                    chunks.append({
                        "header": current_chunk["header"],
                        "h1_header": current_h1,
                        "h2_header": current_h2,
                        "content": "\n".join(current_chunk["content"]).strip()
                    })

                # Update current headers
                if level == 1:
                    current_h1 = title
                    current_h2 = ""
                elif level == 2:
                    current_h2 = title

                current_chunk = {
                    "header": title,
                    "h1": current_h1,
                    "h2": current_h2,
                    "content": [line]
                }
            else:
                if current_chunk["content"] is not None:
                    current_chunk["content"].append(line)

        # Don't forget the last chunk
        if current_chunk["content"]:
            chunks.append({
                "header": current_chunk["header"],
                "h1_header": current_h1,
                "h2_header": current_h2,
                "content": "\n".join(current_chunk["content"]).strip()
            })

        return chunks

    def _preserve_tables(self, text: str) -> str:
        """
        Basic table preservation for text/Markdown tables.
        You can enhance this later with better table detection.
        """
        # Simple regex to detect tables (lines with | or consistent spacing)
        table_pattern = re.compile(r'(?:^.*\|.*$\n?)+', re.MULTILINE)
        tables = table_pattern.findall(text)

        for table in tables:
            if len(table) > 100:  # Only preserve substantial tables
                # You can add logic here to chunk large tables by rows if needed
                pass

        return text  # For now, return as-is. Enhance later.

    def _create_semantic_chunks(self, header_chunks: List[Dict]) -> List[Dict]:
        """
        Use LangChain splitter on each header section for finer chunking + overlap.
        """
        final_chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        for section in header_chunks:
            if not section["content"].strip():
                continue

            # Split content further if it's too large
            sub_chunks = splitter.split_text(section["content"])

            for i, sub_text in enumerate(sub_chunks):
                chunk_text = f"{section['header']}\n\n{sub_text}"

                final_chunks.append({
                    "chunk_text": chunk_text.strip(),
                    "metadata": {
                        "h1_header": section.get("h1_header", ""),
                        "h2_header": section.get("h2_header", ""),
                        "section_header": section.get("header", ""),
                        "chunk_index": i,
                        "source_type": "policy_document"
                    }
                })

        return final_chunks

    def chunk_document(self, file_path: str | Path) -> List[Dict]:
        """
        Main public method: Process one document and return ready-to-embed chunks.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        text = file_path.read_text(encoding="utf-8")

        # Step 1: Split by headers
        header_sections = self._extract_headers_and_content(text)

        # Step 2: Table preservation (can be applied per section)
        for section in header_sections:
            section["content"] = self._preserve_tables(section["content"])

        # Step 3: Create final semantic chunks with overlap
        chunks = self._create_semantic_chunks(header_sections)

        print(f"✅ Chunked {file_path.name} → {len(chunks)} chunks")
        return chunks


# ========================== USAGE EXAMPLE ==========================
if __name__ == "__main__":
    chunker = PolicyChunker(chunk_size=1000, chunk_overlap_percent=0.12)
    
    # Test with one file
    sample_file = "data/raw/travel/sample_travel_policy.txt"
    chunks = chunker.chunk_document(sample_file)
    
    print(f"Total chunks created: {len(chunks)}")
    print("First chunk preview:")
    print(chunks[0]["chunk_text"][:300] + "...")