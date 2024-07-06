---
title: "Llama_index Source Code Analysis(2)"
date: 2024-06-10T14:22:16+08:00
categories: ["llm"]
summary: "This blog introduces the basic concepts of **RAG** and further demonstrates the RAG process based on the source code interpretation of **llama_index**, including data loader, transformation, index, query, etc. In addition, this paper also analyzes the **performance** of llama_index RAG process and gives corresponding optimization suggestions."
---

## Summary

This blog introduces the basic concepts of **RAG** and further demonstrates the RAG process based on the source code interpretation of **llama_index**, including data loader, transformation, index, query, etc. In addition, this paper also analyzes the **performance** of llama_index RAG process and gives corresponding optimization suggestions.

## 4. Llama index Source Code Analysis

### 4.4 Storing

After constructing the index, we store it to avoid redundant constructions. By default, it is stored in memory, but we can also persist it to disk or a database.

#### 4.4.1 Disk

```py
# save
index.storage_context.persist(persist_dir="<persist_dir>")

# load
from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")
index = load_index_from_storage(storage_context)
```

Storing to disk is straightforward, which uses Python’s built-in file handler. It does not employ multiprocessing or multithreading mechanisms. **Note that everything is stored as JSON, which incurs additional overhead.**

```py
# llama_index/core/storage/kvstore/simple_kvstore.py
class SimpleKVStore(BaseInMemoryKVStore):
    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        # ...
        dirpath = os.path.dirname(persist_path)
        with fs.open(persist_path, "w") as f:
            f.write(json.dumps(self._data))
```

Constructing the storage context involves:

```py
# llama_index/core/storage/storage_context.py
class StorageContext:
    @classmethod
    def from_defaults(
        cls, persist_dir: Optional[str] = None, # ...
    ) -> "StorageContext":

        if persist_dir is None:
            # ...
        else:
            docstore = docstore or SimpleDocumentStore.from_persist_dir(
                persist_dir, fs=fs
            )
            index_store = index_store or SimpleIndexStore.from_persist_dir(
                persist_dir, fs=fs
            )
            # ...

            if vector_store:
                vector_stores = {DEFAULT_VECTOR_STORE: vector_store}
            elif vector_stores:
                vector_stores = vector_stores
            else:
                vector_stores = SimpleVectorStore.from_namespaced_persist_dir(
                    persist_dir, fs=fs
                )
            # ...

        return cls(
            docstore=docstore, index_store=index_store, vector_stores=vector_stores, # ...
        )

# llama_index/core/storage/kvstore/simple_kvstore.py
class SimpleKVStore(BaseInMemoryKVStore):
    @classmethod
    def from_persist_path(
        cls, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> "SimpleKVStore":
        fs = fs or fsspec.filesystem("file")
        with fs.open(persist_path, "rb") as f:
            data = json.load(f)
        return cls(data)
```

For each store (including docstore, index_store, graph_store, property_graph_store, vector_stores), which follows a similar process. All file handling is done using `fsspec` with performance comparable to native Python. One potential optimization is converting JSON to binary format.

Then let's see the process of constructing index from storage index:

```py
# llama_index/core/indices/loading.py
def load_index_from_storage(
    storage_context: StorageContext, index_id: Optional[str] = None, # ...
) -> BaseIndex:
    # ...
    indices = load_indices_from_storage(storage_context, index_ids=index_ids, **kwargs)
    return indices[0]
    
def load_indices_from_storage(
    storage_context: StorageContext, index_ids: Optional[Sequence[str]] = None, # ...
) -> List[BaseIndex]:
    if index_ids is None:
        index_structs = storage_context.index_store.index_structs()
    else:
        # ...

    indices = []
    for index_struct in index_structs:
        type_ = index_struct.get_type()    # Vector_store
        index_cls = INDEX_STRUCT_TYPE_TO_INDEX_CLASS[type_]
        index = index_cls(
            index_struct=index_struct, storage_context=storage_context, **kwargs
        )
        indices.append(index)
    return indices

# llama_index/core/storage/index_store/keyval_index_store.py
class KVIndexStore(BaseIndexStore):
    def index_structs(self) -> List[IndexStruct]:
        jsons = self._kvstore.get_all(collection=self._collection)
        return [json_to_index_struct(json) for json in jsons.values()]
```

From a performance perspective, storing to disk incurs minimal overhead, about 0.05s for our example file, resulting in a 452KB file. However, reading takes longer; **reading this small file and constructing the storage context takes 0.12s, and rebuilding the index takes 0.36s**.

The codes above essentially converts the read `storage_context` to a `VectorStoreIndex` class (by pre-constructing the `index_struct`, bypassing the previous embedding construction in `build_index_from_nodes`). In theory, this should not incur significant performance overhead. So where does the 0.36s come from?

Step-by-step investigation reveals that this overhead comes from constructing the index class's `transformations_from_settings_or_context`, specifically from initializing `SentenceSplitter` and calling `get_tokenizer()`.

In `get_tokenizer`, it uses the tiktoken Python library to get a GPT-3.5 token model: `enc = tiktoken.encoding_for_model("gpt-3.5-turbo")`

The `SentenceSplitter` only uses this tokenizer to assess token length:

```py
# llama_index/core/node_parser/text/sentence.py
class SentenceSplitter(MetadataAwareTextSplitter):
    def __init__(
        self, # ...
    ):
        # ...
        self._tokenizer = tokenizer or get_tokenizer()

    def _token_size(self, text: str) -> int:
        return len(self._tokenizer(text))
```

Here, we identify two performance optimization points:

- Lazy loading: The tokenizer should only load when needed, not during every index construction.
- Use a faster method to get the length instead of fully tokenizing the text (frequently used in splitting).

#### 4.4.2 Database

For database storage, we use the officially recommended `chromadb`.

```py
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
documents = SimpleDirectoryReader("./data").load_data()

pipeline = IngestionPipeline(transformations=[SentenceSplitter()])
nodes = pipeline.run(documents=documents)
index = VectorStoreIndex(nodes, storage_context=storage_context)
```

Using it is similar to constructing a conventional index, with the main difference being the additional construction and passing of the storage context to the index.

Internally, the process is almost identical to the in-memory store, with differences in the `add` methods of different stores.

The `ChromaVectorStore` add method:

```py
# llama_index/vector_stores/chroma/base.py
class ChromaVectorStore(BasePydanticVectorStore):
    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        # ...
        for node_chunk in node_chunks:
            # ...
            for node in node_chunk:
                embeddings.append(node.get_embedding())
                metadata_dict = node_to_metadata_dict(
                    node, remove_text=True, flat_metadata=self.flat_metadata
                )
                for key in metadata_dict:
                    if metadata_dict[key] is None:
                        metadata_dict[key] = ""
                metadatas.append(metadata_dict)
                ids.append(node.node_id)
                documents.append(node.get_content(metadata_mode=MetadataMode.NONE))

            self._collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )
            all_ids.extend(ids)

        return all_ids
```

Here, data is organized to lists and passed to the `add` method of the third-party library chromadb, ultimately persisting it to the database.

Note: **ChromaDB** is a high-performance vector database optimized with indexing structures and retrieval algorithms like HNSW (Hierarchical Navigable Small World) and IVF (Inverted File Index), making it more efficient than user-written SQL.

Then let's see the reading process from the database:

```py
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)
```

Initially, the third-party library `chromadb` is used to get the database handler, and then `ChromaVectorStore` initializes the `vector_store` from the handler, constructing the `StorageContext`.

```py
# llama_index/core/storage/storage_context.py
class StorageContext:
    @classmethod
    def from_defaults(
        cls, # ...
    ) -> "StorageContext":
        if persist_dir is None:
            docstore = docstore or SimpleDocumentStore()
            index_store = index_store or SimpleIndexStore()
            # ...

            if vector_store:
                vector_stores = {DEFAULT_VECTOR_STORE: vector_store}
            else:
                vector_stores = vector_stores or {
                    DEFAULT_VECTOR_STORE: SimpleVectorStore()
                }
            if image_store:
                # append image store to vector stores
                vector_stores[IMAGE_VECTOR_STORE_NAMESPACE] = image_store
        else:
            # ...

        return cls(
            docstore=docstore, index_store=index_store, vector_stores=vector_stores,  # ...
        )
```

Unlike reading from disk, here without a `persistent_dir`, a different branch constructs `SimpleDocumentStore`, `SimpleIndexStore`, etc.

After constructing `storage_context`, the key part is `from_vector_store`:

```py
# llama_index/core/indices/vector_store/base.py
class VectorStoreIndex(BaseIndex[IndexDict]):
    @classmethod
    def from_vector_store(
        cls, vector_store: BasePydanticVectorStore, # ...
    ) -> "VectorStoreIndex":
        # ...
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return cls(
            nodes=[],
            embed_model=embed_model,
            service_context=service_context,
            storage_context=storage_context,
            **kwargs,
        )
```

Here, llama_index reconstructs the `StorageContext`, **making the external `StorageContext` useless, which contradicts the documentation**, highlighting some architectural confusion in llama_index.

Then, the class is initialized similarly to constructing from a class, but nodes are empty, meaning no construction occurs. **The actual nodes are fetched from the database during query** (detailed in retrieval).

#### 4.4.3 Insert Document

```py
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex([])
for doc in documents:
    index.insert(doc)

# llama_index/core/indices/base.py
class BaseIndex(Generic[IS], ABC):
    def insert(self, document: Document, **insert_kwargs: Any) -> None:
        with self._callback_manager.as_trace("insert"):
            nodes = run_transformations(
                [document],
                self._transformations, # ...
            )
            self.insert_nodes(nodes, **insert_kwargs)
            self.docstore.set_document_hash(document.get_doc_id(), document.hash)
```

`insert_nodes` includes transformation, ultimately calling `_add_nodes_to_index`, the same as during index initialization.

#### 4.4.4 Delete Document

```py
index = VectorStoreIndex(documents)
index.delete_ref_doc(documents[0].id_)

# llama_index/core/indices/vector_store/base.py
class VectorStoreIndex(BaseIndex[IndexDict]):
    def delete_ref_doc(
        self, ref_doc_id: str, # ...
    ) -> None:
        self._vector_store.delete(ref_doc_id, **delete_kwargs)

        # delete from index_struct only if needed
        if not self._vector_store.stores_text or self._store_nodes_override:
            ref_doc_info = self._docstore.get_ref_doc_info(ref_doc_id)
            if ref_doc_info is not None:
                for node_id in ref_doc_info.node_ids:
                    self._index_struct.delete(node_id)
                    self._vector_store.delete(node_id)

        # delete from docstore only if needed
        if (
            not self._vector_store.stores_text or self._store_nodes_override
        ) and delete_from_docstore:
            self._docstore.delete_ref_doc(ref_doc_id, raise_error=False)

        self._storage_context.index_store.add_index_struct(self._index_struct)
```

`delete` calls the corresponding `vector_store` method, such as `SimpleVectorStore`:

```py
# llama_index/core/vector_stores/simple.py
class SimpleVectorStore(BasePydanticVectorStore):
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        text_ids_to_delete = set()
        for text_id, ref_doc_id_ in self.data.text_id_to_ref_doc_id.items():
            if ref_doc_id == ref_doc_id_:
                text_ids_to_delete.add(text_id)

        for text_id in text_ids_to_delete:
            del self.data.embedding_dict[text_id]
            del self.data.text_id_to_ref_doc_id[text_id]
            if self.data.metadata_dict is not None:
                self.data.metadata_dict.pop(text_id, None)
```

It's worth noting that: this does not directly call the delete interface on `_storage_context.index_store` but instead updates the old key with the new `_index_struct` to delete it. This part does not have significant performance overhead.

For chroma db, it performs actual deletion: `self._collection.delete(where={"document_id": ref_doc_id})`

For disk, since the data is already read into memory, **only in-memory data is deleted, not affecting the disk**.

### 4.5 quering

Once the index is stored, querying can commence.

```py
query_engine = index.as_query_engine()
response = query_engine.query(
    "Write an email to the user given their background information."
)

# llama_index/core/indices/base.py
class BaseIndex(Generic[IS], ABC):
    def as_query_engine(
        self, llm: Optional[LLMType] = None, **kwargs: Any
    ) -> BaseQueryEngine:
        # ...
        retriever = self.as_retriever(**kwargs)
        llm = (
            resolve_llm(llm, callback_manager=self._callback_manager)
            if llm else llm_from_settings_or_context(Settings, self.service_context)
        )
        return RetrieverQueryEngine.from_args(retriever, llm=llm, **kwargs)
```

`as_query_engine` sets up the retriever, selects the LLM (e.g., from settings), and returns a `RetrieverQueryEngine`. This step also initializes `response_synthesizer`, `prompt_helper`, and other components, and this construction is not particularly time-consuming.

Then we query:

```py
# llama_index/core/base/base_query_engine.py
class BaseQueryEngine(ChainableMixin, PromptMixin):
    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        # ...
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        query_result = self._query(str_or_query_bundle)
        # ...
        return query_result
 
# llama_index/core/query_engine/retriever_query_engine.py
class RetrieverQueryEngine(BaseQueryEngine):
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        # ...
        nodes = self.retrieve(query_bundle)
        response = self._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
        )
        return response
```

#### 4.5.1 Retrieval

In a query, we first retrieve the most relevant documents to the user's query.

```py
# llama_index/core/query_engine/retriever_query_engine.py
class RetrieverQueryEngine(BaseQueryEngine):
    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

# llama_index/core/base/base_retriever.py
class BaseRetriever(ChainableMixin, PromptMixin):
    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        nodes = self._retrieve(query_bundle)
        nodes = self._handle_recursive_retrieval(query_bundle, nodes)
        # ...
        return nodes
```

In `_retrieve`, the `get_agg_embedding_from_queries` processes all queries, converting each query to an embedding via `def _embed`. The embeddings are then aggregated using a **mean** reduction to avoid high computation costs for comparing each query.

Next, `self._get_nodes_with_embeddings` retrieves relevant nodes. The query uses different methods depending on the vector store. By default, `SimpleVectorStore` fetches nodes from memory.

```py
# llama_index/core/indices/vector_store/retrievers/retriever.py
class VectorIndexRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                query_bundle.embedding = (
                    self._embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
        return self._get_nodes_with_embeddings(query_bundle)

    def _get_nodes_with_embeddings(
        self, query_bundle_with_embeddings: QueryBundle
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(query_bundle_with_embeddings)
        query_result = self._vector_store.query(query, **self._kwargs)
        return self._build_node_list_from_query_result(query_result)
 
# llama_index/core/vector_stores/simple.py
class SimpleVectorStore(BasePydanticVectorStore):
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        # ...
        if query.mode in LEARNER_MODES:
            # ...
        elif query.mode == MMR_MODE:
            # ...
        elif query.mode == VectorStoreQueryMode.DEFAULT:
            top_similarities, top_ids = get_top_k_embeddings(
                query_embedding, embeddings, # ...
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)
```

Here we take the embedding of the query and the embedding of nodes for similarity calculation, by default, use `get_top_k_embeddings` to take the top-k nodes

```py
# llama_index/core/indices/query/embedding_utils.py
def get_top_k_embeddings(
    query_embedding: List[float], embeddings: List[List[float]], # ...
) -> Tuple[List[float], List]:
    # ...
    similarity_heap: List[Tuple[float, Any]] = []
    for i, emb in enumerate(embeddings_np):
        similarity = similarity_fn(query_embedding_np, emb)
        if similarity_cutoff is None or similarity > similarity_cutoff:
            heapq.heappush(similarity_heap, (similarity, embedding_ids[i]))
            if similarity_top_k and len(similarity_heap) > similarity_top_k:
                heapq.heappop(similarity_heap)
    result_tups = sorted(similarity_heap, key=lambda x: x[0], reverse=True)

    result_similarities = [s for s, _ in result_tups]
    result_ids = [n for _, n in result_tups]
    return result_similarities, result_ids

# llama_index/core/base/embeddings/base.py
def similarity(
    embedding1: Embedding, embedding2: Embedding, mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    if mode == SimilarityMode.EUCLIDEAN:
        return -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
    elif mode == SimilarityMode.DOT_PRODUCT:
        return np.dot(embedding1, embedding2)
    else:
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm
```

Three similarity algorithms are available:

1. **Negative Euclidean Distance** (smaller distance indicates higher similarity)
2. **Dot Product** (higher value indicates higher similarity)
3. **Cosine Similarity** (higher value indicates higher similarity)

After computing similarities, a **min-heap sort** retrieves the top-k results (`NodeWithScore`). This involves complex Python calculations, which could be optimized using C++.

The nodes are then passed to `_handle_recursive_retrieval` to handle potential `IndexNode`s (from user constructs, use scenarios such as recursive document structure, looking at summaries before subdocuments, etc., see [Usage notes](https://docs.llamaindex.ai/en/stable/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval/?h=index_node)).

If using a database like ChromaDB, `ChromaVectorStore` is built. `self._query` directly uses ChromaDB’s API to find top-k similarities.

```py
# llama_index/vector_stores/chroma/base.py
class ChromaVectorStore(BasePydanticVectorStore):
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        # ...
        if not query.query_embedding:
            return self._get(limit=query.similarity_top_k, where=where, **kwargs)

        return self._query(
            query_embeddings=query.query_embedding,
            n_results=query.similarity_top_k,
            # ...
        )
```

#### 4.5.2 Postprocessing

Postprocessing can involve reranking, transformation, or filtering retrieved nodes to include specific metadata (e.g., keywords).

`_apply_node_postprocessors` is called within `retrieve`.

```py
# llama_index/core/query_engine/retriever_query_engine.py
class RetrieverQueryEngine(BaseQueryEngine):
    def _apply_node_postprocessors(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes
```

By default, `_node_postprocessors` is not set, allowing users to customize additional operations. For instance:

```py
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)], # similarity cutoff
)
```

#### 4.5.3 Response synthesis

Combining the query, the most relevant data, and the prompt, the system queries the LLM for a response.

```py
# llama_index/core/response_synthesizers/base.py
class BaseSynthesizer(ChainableMixin, PromptMixin):
    def synthesize(
        self, query: QueryTextType, nodes: List[NodeWithScore], # ...
    ) -> RESPONSE_TYPE:
        # ...
        response_str = self.get_response(
            query_str=query.query_str,
            text_chunks=[
                n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes
            ],
            **response_kwargs,
        )
        # ...
        response = self._prepare_response_output(response_str, source_nodes)
        return response
```

The context includes all content retrieved in the top-k retrieval. If the **context is too large**, adjust the top-k parameter or increase the model’s allowed token limit.

`self.get_response` calls the specific subclass implementation. The default response mode is `COMPACT`, `CompactAndRefine` is invoked.

In `CompactAndRefine`, `_make_compact_text_chunks` uses `query_str` to generate `text_qa_template` and `refine_template`.

```py
# llama_index/core/response_synthesizers/compact_and_refine.py
class CompactAndRefine(Refine):
    @dispatcher.span
    def get_response(
        self, query_str: str, text_chunks: Sequence[str], # ...
    ) -> RESPONSE_TEXT_TYPE:
        new_texts = self._make_compact_text_chunks(query_str, text_chunks)
        return super().get_response(
            query_str=query_str, text_chunks=new_texts, # ...
        )

    def _make_compact_text_chunks(
        self, query_str: str, text_chunks: Sequence[str]
    ) -> List[str]:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        refine_template = self._refine_template.partial_format(query_str=query_str)

        max_prompt = get_biggest_prompt([text_qa_template, refine_template])
        return self._prompt_helper.repack(max_prompt, text_chunks)
```

For example, querying "What did the author do growing up?" generates a `text_qa_template` like:

```bash
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: What did the author do growing up?
Answer:

# conditional template (if chat model used)
"You are an expert Q&A system that strictly operates in two modes when refining existing answers:\n"
# ...
```

`refine_template`:

```bash
The original query is as follows: What did the author do growing up?
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
{context_msg}
------------
Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.
Refined Answer:
```

`llama_index` then calls `get_biggest_prompt` to select the longest template (`refine_template`), and performs `repack`.

```py
# llama_index/core/indices/prompt_helper.py
class PromptHelper(BaseComponent):
    def repack(
        self, prompt: BasePromptTemplate, text_chunks: Sequence[str], # ...
    ) -> List[str]:
        text_splitter = self.get_text_splitter_given_prompt(
            prompt, padding=padding, llm=llm
        )
        combined_str = "\n\n".join([c.strip() for c in text_chunks if c.strip()])
        return text_splitter.split_text(combined_str)
```

`get_text_splitter_given_prompt` gets a new `TokenTextSplitter`, which is used to re-split the context

Then `super().get_response` is then called, where `self._give_response_single` regenerates `text_qa_template` and repacks before invoking `model`, which is quite strange.

```py
# llama_index/core/response_synthesizers/refine.py
class Refine(BaseSynthesizer):
    def get_response(
        self, query_str: str, text_chunks: Sequence[str], # ...
    ) -> RESPONSE_TEXT_TYPE:
        # ...
        for text_chunk in text_chunks:
            if prev_response is None:
                response = self._give_response_single(
                    query_str, text_chunk, **response_kwargs
                )
            else:
                response = self._refine_response_single(
                    prev_response, query_str, text_chunk, **response_kwargs
                )
            prev_response = response
        # ...
        return response
        
def _give_response_single(
        self, query_str: str, text_chunk: str, **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        text_chunks = self._prompt_helper.repack(text_qa_template, [text_chunk])
        # ...
        for cur_text_chunk in text_chunks:
            query_satisfied = False
            if response is None and not self._streaming:
                structured_response = cast(
                    StructuredRefineResponse,
                    program(
                        context_str=cur_text_chunk,
                        **response_kwargs,
                    ),
                )
                if structured_response.query_satisfied:
                    response = structured_response.answer
                # ...
            elif response is None and self._streaming:
                # ...
        # ...
        return response
```

This process highlights a potential design improvement: **regenerating the same template during synthesis can be optimized**.

```py
# llama_index/core/llms/llm.py
class LLM(BaseLLM):
    def predict(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        # ...
        self._log_template_data(prompt, **prompt_args)

        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = self.chat(messages)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            response = self.complete(formatted_prompt, formatted=True)
            output = response.text
        parsed_output = self._parse_output(output)
        return parsed_output
```

`_get_messages` combines the prompt and all context into a complete message, which is then sent to the LLM.

```bash
[
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "You are an expert Q&A system that is trusted around the world.\n"
            "Always answer the query using the provided context information, and not prior knowledge.\n"
            "Some rules to follow:\n"
            "1. Never directly reference the given context in your answer.\n"
            "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines."
        ),
        additional_kwargs={}
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "file_path: /Users/yewentao/Desktop/llama_index/data/paul_graham_essay.txt\n\n"
            "What I Worked On\n\n"
            # ...
            "Computer Science is an uneasy alliance between two halves, theory and systems. The theory people prove things, and the systems people build things.\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\n"
            "Query: What did the author do growing up?\n"
            "Answer: "
        ),
        additional_kwargs={}
    )
]
```

For example, request with Ollama's llama3:

```py
# llama_index/llms/ollama/base.py
class Ollama(CustomLLM):
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # ...
        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=f"{self.base_url}/api/chat",
                json=payload,
            )
            # ...
            return ChatResponse(
                message=ChatMessage(
                    # ...
                ),
            )
```

The response is returned after the model prediction.

From a performance perspective, querying mainly incurs overhead from accessing the LLM and converting the query to embeddings and averaging them. **Optimizing this part further is challenging**.

```bash
get agg embedding time 0.21804475784301758
get nodes embedding time 0.002117156982421875
post llm use time:  1.2400257587432861
query time:  1.4748468399047852
```

Complex Python calculations primarily occur in:

- **Getting node embeddings (comparing all nodes for similarity and top-k)**, which has linear growth in costs. The larger the document, the greater the consumption. For instance, a tenfold increase in document size results in a tenfold increase in time.
- Multiple token splits reappear in queries (as detailed in transformation). However, this accounts for a small proportion due to the selected context.

## 5. Reference

- [llama_index_doc](https://docs.llamaindex.ai/en/stable/)
- [llama_index_github](https://github.com/run-llama/llama_index)
