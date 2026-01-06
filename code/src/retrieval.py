"""
Retrieval Module: RAG with BM25, Embedding Search, and Wikipedia
Enhanced with VLM-based keyword generation for Wikipedia search
Supports loading pre-built vector database for faster initialization
Supports multiple file structures for vector DB (folder or suffix-based)
"""
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
import re

from .config import (
    KB_JSON_PATH, RETRIEVAL_METHOD, TOP_K_RETRIEVE,
    BM25_K1, BM25_B, WIKIPEDIA_LANG, WIKIPEDIA_FALLBACK,
    VIETNAMESE_EMBEDDING_MODEL, VECTOR_DB_PATH,
    OFFLINE_SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)

class RetrievalModule:
    """Module for knowledge retrieval using BM25, Embedding (FAISS), and Wikipedia"""
    
    def __init__(
        self,
        kb_path: Optional[Path] = None,
        method: str = RETRIEVAL_METHOD,
        top_k: int = TOP_K_RETRIEVE,
        vision_module: Optional[object] = None,
        vector_db_path: Optional[Path] = VECTOR_DB_PATH,
        use_prebuilt_index: bool = True
    ):
        """
        Initialize retrieval module
        
        Args:
            kb_path: Path to knowledge base JSON file (default from config)
            method: Retrieval method ("bm25", "embedding", "hybrid")
            top_k: Number of top results to retrieve
            vision_module: Optional VisionModule instance for VLM-based keyword generation
            vector_db_path: Path to pre-built vector database (FAISS index, default from config)
            use_prebuilt_index: Whether to use pre-built FAISS index instead of creating new embeddings
        """
        self.method = method
        self.top_k = top_k
        self.kb_path = Path(kb_path) if kb_path else KB_JSON_PATH
        self.vision_module = vision_module  # VLM for keyword generation
        self.vector_db_path = Path(vector_db_path) if vector_db_path else VECTOR_DB_PATH
        self.use_prebuilt_index = use_prebuilt_index
        
        # Initialize retrieval components
        self.bm25 = None
        self.embedding_model = None
        self.embeddings = None
        self.faiss_index = None  # For pre-built vector database
        
        # Load knowledge base (try pickle metadata first, fallback to JSON)
        self.kb_data = self._load_knowledge_base()
        
        if method in ["bm25", "hybrid"]:
            self._init_bm25()
        
        if method in ["embedding", "hybrid"]:
            self._init_embedding()
        
        # Initialize Wikipedia
        if WIKIPEDIA_FALLBACK:
            self._init_wikipedia()
        else:
            self.wikipedia = None
    
    def _find_index_file(self) -> Optional[Path]:
        """
        Find FAISS index file, supporting multiple naming conventions:
        1. {folder}/vector_db.index (folder structure)
        2. {path}.index (suffix structure)
        3. Any .index file in the folder
        """
        if self.vector_db_path is None:
            return None
        
        # Option 1: Folder structure - {folder}/vector_db.index
        if self.vector_db_path.is_dir():
            index_in_folder = self.vector_db_path / "vector_db.index"
            if index_in_folder.exists():
                return index_in_folder
            # Search for any .index file in folder
            idx_files = list(self.vector_db_path.glob("*.index"))
            if idx_files:
                return idx_files[0]
        
        # Option 2: Suffix structure - {path}.index
        suffix_index = Path(str(self.vector_db_path) + ".index")
        if suffix_index.exists():
            return suffix_index
        
        # Option 3: Search parent directory for .index files
        parent = self.vector_db_path.parent
        if parent.exists():
            idx_files = list(parent.glob("*.index"))
            if idx_files:
                return idx_files[0]
        
        return None
    
    def _find_metadata_file(self) -> Optional[Path]:
        """
        Find metadata file, supporting multiple formats:
        1. {folder}/vector_db_config.json (folder structure)
        2. {path}_metadata.pkl (pickle structure)
        3. Any .pkl or config.json file in the folder
        """
        if self.vector_db_path is None:
            return None
        
        # Option 1: Folder structure - config JSON
        if self.vector_db_path.is_dir():
            config_json = self.vector_db_path / "vector_db_config.json"
            if config_json.exists():
                return config_json
        
        # Option 2: Suffix structure - pickle metadata
        suffix_pkl = Path(str(self.vector_db_path) + "_metadata.pkl")
        if suffix_pkl.exists():
            return suffix_pkl
        
        # Option 3: Search for any .pkl file
        if self.vector_db_path.is_dir():
            pkl_files = list(self.vector_db_path.glob("*.pkl"))
            if pkl_files:
                return pkl_files[0]
        
        parent = self.vector_db_path.parent
        if parent.exists():
            pkl_files = list(parent.glob("*.pkl"))
            if pkl_files:
                return pkl_files[0]
        
        return None
    
    def _load_knowledge_base(self) -> List[Dict]:
        """
        Load knowledge base - try pickle metadata first (faster), fallback to JSON
        """
        # Try loading from pickle metadata first (optimized for pre-built index)
        if self.use_prebuilt_index:
            meta_file = self._find_metadata_file()
            if meta_file and meta_file.suffix == '.pkl':
                try:
                    logger.info(f"Loading metadata from pickle: {meta_file}")
                    with open(meta_file, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"Loaded {len(data)} entries from pickle metadata")
                    return data
                except Exception as e:
                    logger.warning(f"Failed to load pickle metadata: {e}")
        
        # Fallback: Load from JSON
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} entries from knowledge base JSON")
            return data
        except FileNotFoundError:
            logger.warning(f"Knowledge base not found at {self.kb_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return []
    
    def _init_bm25(self):
        """Initialize BM25 retriever"""
        try:
            from rank_bm25 import BM25Okapi
            from underthesea import word_tokenize
            
            # Tokenize all documents
            self.tokenized_docs = []
            for item in self.kb_data:
                # Combine entity, facts, summary into searchable text
                text = f"{item.get('entity', '')} {item.get('facts', '')} {item.get('summary', '')}"
                tokens = word_tokenize(text.lower())
                self.tokenized_docs.append(tokens)
            
            if self.tokenized_docs:
                self.bm25 = BM25Okapi(
                    self.tokenized_docs,
                    k1=BM25_K1,
                    b=BM25_B
                )
                logger.info("BM25 initialized")
            else:
                logger.warning("No documents for BM25")
                
        except ImportError:
            logger.warning("rank_bm25 or underthesea not available")
        except Exception as e:
            logger.error(f"Error initializing BM25: {e}")
    
    def _init_embedding(self):
        """Initialize embedding model and load/create embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try to load pre-built vector database first
            if self.use_prebuilt_index and self._load_prebuilt_vector_db():
                logger.info("Using pre-built vector database")
                # Only load embedding model for query encoding (not for creating KB embeddings)
                logger.info(f"Loading embedding model for queries: {VIETNAMESE_EMBEDDING_MODEL}")
                self.embedding_model = SentenceTransformer(VIETNAMESE_EMBEDDING_MODEL)
                return
            
            # Fallback: create embeddings from scratch
            logger.info("Pre-built index not found. Creating embeddings from scratch...")
            logger.info(f"Loading embedding model: {VIETNAMESE_EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(VIETNAMESE_EMBEDDING_MODEL)
            
            # Create embeddings for all documents
            if self.kb_data:
                texts = []
                for item in self.kb_data:
                    text = f"{item.get('entity', '')} {item.get('facts', '')} {item.get('summary', '')}"
                    texts.append(text)
                
                logger.info(f"Creating embeddings for {len(texts)} knowledge base entries...")
                self.embeddings = self.embedding_model.encode(
                    texts,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                logger.info("Embeddings created successfully (fallback mode)")
            else:
                self.embeddings = None
                logger.warning("No documents to create embeddings for")
                
        except ImportError:
            logger.warning("sentence-transformers not available")
        except Exception as e:
            logger.error(f"Error initializing embedding: {e}")
    
    def _load_prebuilt_vector_db(self) -> bool:
        """
        Load pre-built FAISS vector database
        Supports multiple file structures (folder-based or suffix-based)
        
        Returns:
            True if successfully loaded, False otherwise
        """
        index_path = self._find_index_file()
        
        if index_path is None:
            logger.info(f"No pre-built FAISS index found for {self.vector_db_path}")
            return False
        
        try:
            import faiss
            
            # Load optional config (for logging purposes)
            meta_file = self._find_metadata_file()
            if meta_file and meta_file.suffix == '.json':
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    logger.info(f"Loading pre-built vector database:")
                    logger.info(f"  - Model: {config.get('embedding_model', 'unknown')}")
                    logger.info(f"  - Dimension: {config.get('dimension', 'unknown')}")
                    logger.info(f"  - Num vectors: {config.get('num_vectors', 'unknown')}")
                except Exception:
                    pass
            
            # Load FAISS index
            logger.info(f"Loading FAISS index from: {index_path}")
            self.faiss_index = faiss.read_index(str(index_path))
            logger.info(f"FAISS index loaded successfully with {self.faiss_index.ntotal} vectors")
            
            return True
            
        except ImportError:
            logger.warning("faiss not available, falling back to creating embeddings")
            return False
        except Exception as e:
            logger.error(f"Error loading pre-built vector database: {e}")
            return False
    
    def _init_wikipedia(self):
        """Initialize Wikipedia search"""
        try:
            import wikipedia
            wikipedia.set_lang(WIKIPEDIA_LANG)
            self.wikipedia = wikipedia
            logger.info(f"Wikipedia initialized for language: {WIKIPEDIA_LANG}")
        except ImportError:
            logger.warning("wikipedia package not available")
            self.wikipedia = None
        except Exception as e:
            logger.error(f"Error initializing Wikipedia: {e}")
            self.wikipedia = None
    
    def _bm25_search(self, query: str) -> List[Tuple[int, float]]:
        """Search using BM25"""
        if self.bm25 is None:
            return []
        
        try:
            from underthesea import word_tokenize
            query_tokens = word_tokenize(query.lower())
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:self.top_k]
            results = [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]
            return results
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def _embedding_search(self, query: str) -> List[Tuple[int, float]]:
        """Search using embeddings (supports both FAISS index and numpy arrays)"""
        if self.embedding_model is None:
            return []
        
        # Check if we have neither FAISS index nor numpy embeddings
        if self.faiss_index is None and self.embeddings is None:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True
            )
            
            # Use FAISS index if available (pre-built database)
            if self.faiss_index is not None:
                return self._faiss_search(query_embedding)
            
            # Fallback: use numpy arrays
            return self._numpy_embedding_search(query_embedding)
            
        except Exception as e:
            logger.error(f"Error in embedding search: {e}")
            return []
    
    def _faiss_search(self, query_embedding: np.ndarray) -> List[Tuple[int, float]]:
        """Search using FAISS index with L2 normalization"""
        try:
            import faiss
            
            # Reshape for FAISS (expects 2D array)
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            
            # Normalize query vector for cosine similarity (if index was built with normalized vectors)
            faiss.normalize_L2(query_vector)
            
            # Search
            distances, indices = self.faiss_index.search(query_vector, self.top_k)
            
            # Convert to list of (idx, score) tuples
            # For normalized vectors with inner product, distance = cosine similarity
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if idx >= 0:  # FAISS returns -1 for invalid results
                    results.append((int(idx), float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")
            return []
    
    def _numpy_embedding_search(self, query_embedding: np.ndarray) -> List[Tuple[int, float]]:
        """Search using numpy array embeddings (cosine similarity)"""
        try:
            # Compute cosine similarity
            scores = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:self.top_k]
            results = [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]
            return results
        except Exception as e:
            logger.error(f"Error in numpy embedding search: {e}")
            return []
    
    def _generate_keywords_with_vlm(
        self,
        image,
        question: str,
        caption: str = "",
        ocr_text: str = ""
    ) -> str:
        """
        Returns:
            Generated keywords string for Wikipedia search
        """
        if self.vision_module is None:
            # Fallback: combine question, caption, and OCR
            keywords = f"{question} {caption} {ocr_text}".strip()
            logger.warning("Vision module not available, using fallback keyword generation")
            return keywords
        
        try:
            # Use VLM to generate keywords capturing the meaning
            prompt = (
                f"Dựa vào câu hỏi: '{question}'\n"
                f"Mô tả hình ảnh: {caption}\n"
                f"Văn bản trong ảnh: {ocr_text}\n\n"
                "Hãy tạo ra 3-5 từ khóa quan trọng nhất (tên riêng, địa danh, sự kiện lịch sử, "
                "công trình văn hóa) để tìm kiếm thông tin trên Wikipedia. "
                "Chỉ trả về các từ khóa, cách nhau bằng dấu phẩy, không giải thích thêm."
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Use vision module's processor and model
            text = self.vision_module.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Extract images from messages
            images = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for content in msg["content"]:
                        if content.get("type") == "image":
                            images.append(content["image"])
            
            inputs = self.vision_module.processor(
                text=[text],
                images=images if images else None,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            import torch
            from .config import DEVICE
            # Handle device for models with device_map="auto"
            if hasattr(self.vision_module.model, 'device'):
                device = self.vision_module.model.device
            elif hasattr(self.vision_module.model, 'hf_device_map'):
                # Model is split across devices, use first device or default
                device = DEVICE
            else:
                device = DEVICE
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate keywords
            with torch.no_grad():
                generated_ids = self.vision_module.model.generate(
                    **inputs,
                    max_new_tokens=64,  # Short keywords
                    do_sample=False
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            keywords = self.vision_module.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            logger.info(f"Generated keywords: {keywords}")
            return keywords
            
        except Exception as e:
            logger.error(f"Error generating keywords with VLM: {e}")
            # Fallback to simple combination
            keywords = f"{question} {caption} {ocr_text}".strip()
            return keywords
    
    def _chunk_wikipedia_content(self, content: str, chunk_size: int = 500) -> List[str]:
        """
        Split Wikipedia content into chunks for better retrieval.
        Returns:
            List of content chunks
        """
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size, save current chunk
            if current_chunk and len(current_chunk) + len(para) > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _rank_chunks_by_similarity(
        self,
        chunks: List[str],
        query: str,
        keywords: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Rank chunks by semantic similarity to query and keywords.
        
        Args:
            chunks: List of content chunks
            query: Original question
            keywords: Generated keywords
            top_k: Number of top chunks to return
            
        Returns:
            List of (chunk, score) tuples sorted by score
        """
        if not chunks or self.embedding_model is None:
            # Fallback: return first chunks
            return [(chunk, 1.0) for chunk in chunks[:top_k]]
        
        try:
            # Combine query and keywords for search
            search_text = f"{query} {keywords}".strip()
            
            # Encode search text
            search_embedding = self.embedding_model.encode(
                search_text,
                convert_to_numpy=True
            )
            
            # Encode all chunks
            chunk_embeddings = self.embedding_model.encode(
                chunks,
                convert_to_numpy=True
            )
            
            # Compute cosine similarity
            scores = np.dot(chunk_embeddings, search_embedding) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(search_embedding)
            )
            
            # Get top-k chunks
            top_indices = np.argsort(scores)[::-1][:top_k]
            ranked_chunks = [(chunks[idx], float(scores[idx])) for idx in top_indices]
            
            return ranked_chunks
            
        except Exception as e:
            logger.error(f"Error ranking chunks: {e}")
            # Fallback: return first chunks
            return [(chunk, 1.0) for chunk in chunks[:top_k]]
    
    def _wikipedia_search(
        self,
        query: str,
        image=None,
        caption: str = "",
        ocr_text: str = "",
        max_results: int = 3
    ) -> List[Dict]:
        """
        Search Wikipedia for additional information with VLM-generated keywords.
        Enhanced with chunking and semantic ranking.
        
        Args:
            query: Original question
            image: PIL Image object (optional, for VLM keyword generation)
            caption: Image caption
            ocr_text: OCR text from image
            max_results: Maximum number of Wikipedia pages to retrieve
            
        Returns:
            List of retrieved Wikipedia documents
        """
        if self.wikipedia is None:
            return []
        
        try:
            # Step 1: Generate keywords using VLM if available
            if image is not None and self.vision_module is not None:
                search_keywords = self._generate_keywords_with_vlm(
                    image=image,
                    question=query,
                    caption=caption,
                    ocr_text=ocr_text
                )
                logger.info(f"Using VLM-generated keywords: {search_keywords}")
            else:
                # Fallback: use query + caption + OCR
                search_keywords = f"{query} {caption} {ocr_text}".strip()
                logger.info(f"Using fallback keywords: {search_keywords}")
            
            # Step 2: Search Wikipedia with keywords
            search_results = self.wikipedia.search(search_keywords, results=max_results)
            
            retrieved = []
            for page_title in search_results:
                try:
                    page = self.wikipedia.page(page_title)
                    full_content = page.content
                    
                    # Step 3: Split content into chunks
                    chunks = self._chunk_wikipedia_content(full_content, chunk_size=500)
                    
                    # Step 4: Rank chunks by semantic similarity
                    ranked_chunks = self._rank_chunks_by_similarity(
                        chunks=chunks,
                        query=query,
                        keywords=search_keywords,
                        top_k=2  # Get top 2 chunks per page
                    )
                    
                    # Combine top chunks
                    best_content = "\n\n".join([chunk for chunk, _ in ranked_chunks])
                    
                    # If no chunks ranked, use first 500 chars as fallback
                    if not best_content:
                        best_content = full_content[:500]
                    
                    retrieved.append({
                        "title": page_title,
                        "content": best_content,
                        "url": page.url,
                        "source": "wikipedia",
                        "keywords_used": search_keywords
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing Wikipedia page {page_title}: {e}")
                    continue
            
            return retrieved
            
        except Exception as e:
            logger.error(f"Error in Wikipedia search: {e}")
            return []
    
    def retrieve(
        self,
        query: str,
        caption: str = "",
        ocr_text: str = "",
        image=None,
        offline_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Retrieve relevant knowledge base entries from BOTH offline (KB) and online (Wikipedia).
        Offline results are only included if their similarity score meets the threshold.
        
        Args:
            query: User question
            caption: Image caption
            ocr_text: OCR text from image
            image: PIL Image object (optional, for VLM keyword generation in Wikipedia search)
            offline_threshold: Minimum similarity score for offline results (default from config)
            
        Returns:
            List of retrieved documents with metadata (combined from KB and Wikipedia)
        """
        # Use config threshold if not specified
        if offline_threshold is None:
            offline_threshold = OFFLINE_SIMILARITY_THRESHOLD
        
        # Combine query components
        search_query = f"{query} {caption} {ocr_text}".strip()
        
        # ============ STEP 1: Search Offline (Knowledge Base) ============
        offline_results = []
        raw_scores = []  # Store raw embedding scores for threshold filtering
        
        if self.method == "bm25":
            bm25_results = self._bm25_search(search_query)
            for idx, score in bm25_results:
                if idx < len(self.kb_data):
                    item = self.kb_data[idx].copy()
                    item["score"] = score
                    item["source"] = "kb_bm25"
                    # BM25 scores need normalization - use raw score for now
                    item["raw_score"] = score
                    offline_results.append(item)
                    raw_scores.append(score)
        
        elif self.method == "embedding":
            emb_results = self._embedding_search(search_query)
            for idx, score in emb_results:
                if idx < len(self.kb_data):
                    item = self.kb_data[idx].copy()
                    item["score"] = score
                    item["source"] = "kb_embedding"
                    item["raw_score"] = score  # Embedding score is already normalized (0-1)
                    offline_results.append(item)
                    raw_scores.append(score)
        
        elif self.method == "hybrid":
            # Combine BM25 and embedding results with weighted scoring
            bm25_results = self._bm25_search(search_query)
            emb_results = self._embedding_search(search_query)
            
            # Convert to dicts for easier merging
            bm25_hits = dict(bm25_results)
            emb_hits = dict(emb_results)
            
            # Get all unique indices
            all_indices = set(bm25_hits.keys()) | set(emb_hits.keys())
            
            # Calculate hybrid scores with weighted combination
            # BM25: 30%, Embedding: 70% (scaled by 10 to normalize range)
            hybrid_scores = []
            for idx in all_indices:
                bm25_score = bm25_hits.get(idx, 0.0)
                emb_score = emb_hits.get(idx, 0.0)
                # Weighted combination: prioritize semantic similarity
                final_score = bm25_score * 0.3 + emb_score * 0.7 * 10
                # Store raw embedding score for threshold filtering
                hybrid_scores.append((idx, final_score, emb_score))
            
            # Sort by hybrid score
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Build results
            for idx, score, emb_score in hybrid_scores[:self.top_k * 2]:  # Get more candidates
                if idx < len(self.kb_data):
                    item = self.kb_data[idx].copy()
                    item["score"] = score
                    item["source"] = "kb_hybrid"
                    item["raw_score"] = emb_score  # Use embedding score for threshold
                    offline_results.append(item)
                    raw_scores.append(emb_score)
        
        # ============ STEP 2: Filter offline results by threshold ============
        high_quality_offline = [
            r for r in offline_results 
            if r.get("raw_score", 0) >= offline_threshold
        ]
        
        # Log filtering results
        if offline_results:
            max_score = max(raw_scores) if raw_scores else 0
            logger.info(f"Offline search: {len(offline_results)} candidates, "
                        f"max_score={max_score:.3f}, threshold={offline_threshold}")
            logger.info(f"After filtering: {len(high_quality_offline)} high-quality results")
        
        # ============ STEP 3: Always search Online (Wikipedia) ============
        wiki_results = []
        if WIKIPEDIA_FALLBACK:
            # Calculate how many Wikipedia results we need
            # If offline has high-quality results, get fewer from Wikipedia
            # If offline quality is low, get more from Wikipedia
            wiki_count = max(1, self.top_k - len(high_quality_offline))
            
            wiki_results = self._wikipedia_search(
                query=query,
                image=image,
                caption=caption,
                ocr_text=ocr_text,
                max_results=wiki_count
            )
            logger.info(f"Wikipedia search: {len(wiki_results)} results")
        
        # ============ STEP 4: Combine and rank results ============
        # Priority: High-quality offline > Wikipedia > Low-quality offline
        final_results = []
        
        # Add high-quality offline results first
        for item in high_quality_offline[:self.top_k]:
            # Remove raw_score from final output
            item.pop("raw_score", None)
            final_results.append(item)
        
        # Add Wikipedia results
        remaining_slots = self.top_k - len(final_results)
        for item in wiki_results[:remaining_slots]:
            final_results.append(item)
        
        # If still not enough, add lower-quality offline results
        remaining_slots = self.top_k - len(final_results)
        if remaining_slots > 0:
            low_quality_offline = [
                r for r in offline_results 
                if r.get("raw_score", 0) < offline_threshold
            ]
            for item in low_quality_offline[:remaining_slots]:
                item.pop("raw_score", None)
                item["source"] = item.get("source", "kb") + "_low_conf"
                final_results.append(item)
        
        logger.info(f"Final retrieval: {len(final_results)} results "
                    f"(offline_high={len(high_quality_offline)}, wiki={len(wiki_results)})")
        
        return final_results[:self.top_k]

