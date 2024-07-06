---
title: "Llama_index 源码解读(2)"
date: 2024-06-10T14:22:16+08:00
categories: ["llm"]
summary: "本文介绍了**RAG**的基础概念，并基于**llama_index**的源码解读进一步演示了RAG的流程，包括data loader、transformation、index、query等。此外，本文也对llama_index RAG流程进行了一些性能上的分析并给出了对应优化建议。"
---

## Summary

本文介绍了**RAG**的基础概念，并基于**llama_index**的源码解读进一步演示了RAG的流程，包括data loader、transformation、index、query等。此外，本文也对llama_index RAG流程进行了一些性能上的分析并给出了对应优化建议。

## 1. Introduction

llama_index是一个用于构建上下文增强的大模型应用的框架，即基于用户的私有数据让模型在特定领域有更好表现。

llama_index主要提供以下工具：

- **data connector**：连接到用户私有数据，API、数据库等
- **data indexes**：数据structure成便于llm使用的形式
- **Engines**：提供自然语言访问途径
  - 如query engine：question-answering的接口，如知识库查询等
  - 如chat engine：多轮对话的接口，如gpt
- **Agents**：基于LLM提供服务，如任务自动化，客服等
- **Observability/Evaluation**：集成应用评估、监控工具

调研基于 `llama-index==0.10.40` 版本

## 2. RAG High Level Concepts

RAG即Retrieval-Augmented Generation，检索增强生成

通常，大模型使用公开数据集进行训练，但对于特定任务可能表现不佳。RAG可以将用户私有数据加入到大模型可以访问到的数据中，作为上下文一起喂给模型，这一步骤**不需要对模型进行微调或训练**

![image](rag_process.png)

大致流程：

1. 数据加载和索引：

    - 加载数据：将私有数据（例如文档（非结构化）、数据库（结构化）、API 等）加载。
    - 创建索引：对加载的数据进行预处理和索引，以便快速检索。这个 Index 是一个结构化的中间表示，能够高效地筛选出与查询相关的内容。

2. 用户查询：

    - 首先查询预先创建的索引
    - 检索：从索引中筛选出与查询最相关的内容
    - 这些被检索到的相关内容构成了上下文，用于辅助 LLM 的生成过程。

3. 生成回答：

    - 结合上下文和查询：将检索到的相关内容（上下文）与用户查询一起传递给 LLM。
    - 生成响应：LLM 使用这些上下文信息生成更准确、相关性更高的回答。

![image](rag_process_2.png)

技术上来说，一共有五个阶段：

- **Loading**:

  - **Nodes and Documents**: `Document`是一个容器，如PDF、API等都可以封装为一个容器，包含完整的数据源内容；`Node` 是LlamaIndex 中的数据原子单位，表示源 Document 的一个“块”或片段，每个`Node`都有自己的meta data来与所在的document和其他node产生关联
  - **Connectors**: 或者称之为`Reader`，将数据源structure处理并转化为 `Documents` 和 `Nodes`.

- **Indexing**:

  - **Indexes**: 组织好的数据索引，如存储在`VectorStore`中的`vector embeddings`。索引也包含所需的metadata
  - **Embeddings**：`embeddings`即文本的数值表示。如vector embeddings是高维向量，表示数据的语义信息，相似语义的数据在向量空间中接近，进而便于query

- **Storing**: 把已经构建好的索引和其他metadata存储，避免反复构建

- **Querying**:

  - **Retrievers**: `retriever`定义了在接收到查询时，如何高效地从索引中检索相关的context。检索策略直接影响检索到的数据的相关性和检索效率
  - **Routers**: `router` 决定使用哪个 Retriever 检索相关的context。具体而言，我们使用 `RouterRetriever` 类负责选择一个或多个候选 Retriever 执行查询，使用`selector`根据metadata 和 query 内容决定最佳retriever.
  - **Node Postprocessors**:  接收一组检索到的节点并对其应用变换、过滤或重新排序逻辑。
  - **Response Synthesizers**: 使用用户query和检索context+promot拼接成输入，基于大模型生成response

- **Evaluation**: 评估查询策略、pipeline及结果是否准确。

## 3. Llama index使用示例

我们使用[**ollama**](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/)云端部署了一个7B llama3，documents来自于一个简短的[文本](https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt)，78KB）然后在mac（Core i7 2.6 GHz）上运行以下代码

```py
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

start_time = time.time()

load_start = time.time()
documents = SimpleDirectoryReader("data").load_data()
load_end = time.time()

embed_start = time.time()
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
embed_end = time.time()

llm_start = time.time()
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
llm_end = time.time()

index_start = time.time()
index = VectorStoreIndex.from_documents(documents)
index_end = time.time()

query_engine_start = time.time()
query_engine = index.as_query_engine()
query_engine_end = time.time()

query_start = time.time()
response = query_engine.query("What did the author do growing up?")
query_end = time.time()

print(response)

print(f"Data loading time: {load_end - load_start} seconds")
print(f"Embedding model setup time: {embed_end - embed_start} seconds")
print(f"LLM setup time: {llm_end - llm_start} seconds")
print(f"Index creation time: {index_end - index_start} seconds")
print(f"Query engine creation time: {query_engine_end - query_engine_start} seconds")
print(f"Query execution time: {query_end - query_start} seconds")
print(f"Total time: {time.time() - start_time} seconds")
```

输出：

```bash
According to the provided context, before college, the author worked on writing and programming outside of school. Specifically, he wrote short stories in his teenage years and tried writing programs on an IBM 1401 computer using an early version of Fortran in 9th grade (when he was around 13 or 14).
Data loading time: 0.021808862686157227 seconds
Embedding model setup time: 3.6557559967041016 seconds
LLM setup time: 0.0005099773406982422 seconds
Index creation time: 10.546114921569824 seconds
Query engine creation time: 0.0671701431274414 seconds
Query execution time: 1.3822910785675049 seconds
Total time: 15.673884868621826 seconds
```

可以看到，即使是78KB的一篇document创建index、query到返回结果，总共需要15s左右时间左右时间，其中创建index就花了10s+，为什么会有这么大的时间开销呢？我们后文进一步解析

## 4. llama index源码解析

### 4.4 Storing

在构造好index之后，我们会将其存储下来避免重复构建。默认情况下，存储在in-memory中，我们也可以将其输出到disk或db持久化存储

#### 4.4.1 Disk

```py
# save
index.storage_context.persist(persist_dir="<persist_dir>")

# load
from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")
index = load_index_from_storage(storage_context)
```

存储到磁盘上较为简单，这里不详细展开，使用python自带的file handler存储，也没有使用多进程多线程等机制。**注意这里存储的都是json所以会产生较多额外开销**

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

构造storage context：

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

对于每个store（包含docstore、index_store、graph_store、property_graph_store、vector_stores）都是类似的过程，不再赘述，此处都是python file handler（`fsspec`存读文件与原生python性能相当），性能优化点可能在于将json改成binary

我们再看将storage_index构建为index的过程

```py
# llama_index/core/indices/loading.py
def load_index_from_storage(
    storage_context: StorageContext,
    index_id: Optional[str] = None,
    **kwargs: Any,
) -> BaseIndex:
    # ...
    indices = load_indices_from_storage(storage_context, index_ids=index_ids, **kwargs)
    # ...
    return indices[0]
    
def load_indices_from_storage(
    storage_context: StorageContext,
    index_ids: Optional[Sequence[str]] = None,
    **kwargs: Any,
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

**性能上看**，存储到磁盘开销较小，仅为0.05s，如我们示例中的文件，存储到磁盘后452KB。但读取时间耗时较长，**读这样的小文件构造storage context花费0.12s，而再构建index花费0.36s**

上面的代码本质是将读出来的`storage_context`转换为`vectorstore_index`类（通过提前构建好`index_struct`跳过了之前embedding的构建过程（`build_index_from_nodes`），这里理论上不应该会有大的性能开销，那0.36s是怎么来的呢？

我们一步一步排查，发现这一笔开销来自于构建index类时的`transformations_from_settings_or_context`，进一步说，来自于底下默认`SentenceSplitter`初始化时`get_tokenizer()`的开销

在`get_tokenizer`中，我们发现它使用tiktoken python库拿了一个gpt3.5的token model `enc = tiktoken.encoding_for_model("gpt-3.5-turbo")`

而这个tokenizer在`SentenceSplitter`中主要做什么呢？仅仅用于评估token长度，如

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

在这里我们发现了两个性能优化点：

- 懒加载：在当前场景下完全不需要使用tokenizer，设计上应该只在使用到时加载
- 使用更快的方式获取长度，而不需要完整tokenize后取长度（而这部分在切分中经常使用）

#### 4.4.2 Database

存储到db中，我们使用官方推荐的`chromadb`

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

使用上和常规构建index没有太大区别，主要是额外构建好了storage_context并传入index

在内部构建流程中，与in memory的store也基本一致，区分点在于不同store的不同`add`方法

如调用到`ChromaVectorStore`的add方法：

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

这里将数据整理一下，然后调用`self._collection.add`即到三方库chromadb的`add`方法，最后落盘到db

补充：**ChromaDB**是一种高性能的矢量数据库，具有针对性优化的索引结构和检索算法，如 HNSW（Hierarchical Navigable Small World）图和 IVF（Inverted File Index），这比用户自己写sql往往更高效

我们再看从db中读取的过程

```py
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)
```

一开始直接用三方库chromadb来拿到数据库handler，然后`ChromaVectorStore`用db handler得到的collection初始化得到`vector_store`

随后我们用vector_store构建`StorageContext`

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

和上文从磁盘中读取不同的是，没传`persistent_dir`我们走的是另一个分支从头开始构建`SimpleDocumentStore`、`SimpleIndexStore`等对象

构建好`storage_context`后，我们重点看`from_vector_store`

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

在这里llama_index又构建了一次`StorageContext`，**外面传的StorageContext完全没用，和文档明显冲突**，所以这里确实是显示了llama_index架构比较混乱的问题

然后调用`cls`初始化，和从类里构建index类似，但此时`nodes`是空的，所以相当于没有构建，**之后在query的时候再对应从db里获取**，这部分详见下文retrieval

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

`insert_nodes`内置了transformation，然后一路调用到`_add_nodes_to_index`，之后便与初始化构建index过程相同，不再赘述。

#### 4.4.4 Delete Document

```py
index = VectorStoreIndex(documents)
index.delete_ref_doc(documents[0].id_)

# llama_index/core/indices/vector_store/base.py
class VectorStoreIndex(BaseIndex[IndexDict]):
    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
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

`delete`调用对应`vector_store`的方法，如`SimpleVectorStore`

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

值得指出的是：这里不是直接将`_storage_context.index_store`调用删除接口，而是直接将删除后的`_index_struct` put进键值对里，更新旧键来删除。这部分没有明显有性能开销的地方

如果是chroma db这里会进行真的删除：`self._collection.delete(where={"document_id": ref_doc_id})`

对于disk，则因为已经读进内存了，**所以只删除了内存中的数据，对磁盘没有影响。**

### 4.5 quering

在存储index后，我们就可以开始查询了

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

`as_query_engine`会构造好retriever和选择llm（如settings中拿）然后返回`RetrieverQueryEngine`（在这里又初始化了`response_synthesizer`、`prompt_helper`等，这部分构建没有特别耗时的内容）

然后我们开始query

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

在query中，我们需要先拿用户的query进行检索最具相关性的文档

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

在`_retrieve`中，首先会调用`get_agg_embedding_from_queries`处理所有query，即调用`def _embed`（上文介绍过）来将每一个query转化为embedding，然后所有query agg mean后返回（这里需要mean，否则每个query都比较相关度成本太高）

随后调用`self._get_nodes_with_embeddings`去检索相关节点，这里query会根据vector store不同而选择不同的方法，默认是in memory的`SimpleVectorStore`直接从内存中获取

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

这里我们拿query的embedding和nodes的embedding进行相似度计算，默认情况下，使用`get_top_k_embeddings`拿top-k的nodes

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

这里相似度有三种算法，**负欧几里得距离**（两点直线距离，距离越小相似度越大）、**点积**（值越大相似度越大）、**余弦相似度**（上文介绍过的夹角）。

求出相似度后，利用**小根堆排序**（这里有一点性能问题，`heappushpop`就可以，不用push后pop）求topk返回NodeWithScore。**这里存在较复杂的python计算，或许也可以改成c++来优化**

随后返回调用`_handle_recursive_retrieval`，这里是处理nodes中可能存在的`IndexNode`（来自于用户构造，使用场景如递归文档结构，先看摘要再看子文档等，详见[使用说明](https://docs.llamaindex.ai/en/stable/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval/?h=index_node)

如果是从db中（如chroma db），则调用到`ChromaVectorStore`中，随后直接调用`self._query`调用chroma db提供的查询API寻找topk相似度。

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
            where=where,
            **kwargs,
        )
```

#### 4.5.2 Postprocessing

Postprocessing阶段我们会对检索到retrieve到的nodes进行rerank（重排序）、transformation或filter，如需求node附带特定的metadata（如关键字）。

在`retrieve`中我们调用`_apply_node_postprocessors`进入后处理

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

默认情况下`_node_postprocessors`并没有设置值，主要给用户提供了额外的操作空间。如

```py
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)], # similarity cutoff
)
```

#### 4.5.3 Response synthesis

将我们的query、最相关的数据和提示词一起送给LLM获取回复

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

注意：这里context包含之前topk retrieve检索到的所有内容，如果**context过大需要将topk参数调小或者调大模型允许的token数量**

`self.get_response`会调用到具体subclass的实现，如初始化时默认的response mode为`COMPACT`那么就会调用到`CompactAndRefine`处

`CompactAndRefine`会调用`_make_compact_text_chunks`利用`query_str`生成`text_qa_template`和`refine_template`

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

例如，我们询问了`What did the author do growing up?`

那么就会生成text_qa_template如

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

refine_template如

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

然后llama_index会`get_biggest_prompt`选出最长的template，这里是`refine_template`,然后进行`repack`

```py
# llama_index/core/indices/prompt_helper.py
class PromptHelper(BaseComponent):
    def repack(
        self,
        prompt: BasePromptTemplate,
        text_chunks: Sequence[str],
        padding: int = DEFAULT_PADDING,
        llm: Optional[LLM] = None,
    ) -> List[str]:
        text_splitter = self.get_text_splitter_given_prompt(
            prompt, padding=padding, llm=llm
        )
        combined_str = "\n\n".join([c.strip() for c in text_chunks if c.strip()])
        return text_splitter.split_text(combined_str)
```

`get_text_splitter_given_prompt`拿到一个新的`TokenTextSplitter`然后对context再做切分

然后调用`super().get_response`

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

在`self._give_response_single`中再次生成text_qa_template再次repack，然后program()实际调用model。**此处又生成同样的模板，这里设计上存在改进的空间**

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

`_get_messages`将prompt和所有context拼成完整的message，如我们例子中就得到了

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

`messages`整理好后，向大模型发起请求，我们这里是Ollama的llama3

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

大模型predict完成后，一路返回，query结束

性能上而言，query的大头还是在大模型本身的访问上，随后就是将query转化为embedding并平均的过程，**这部分本身优化空间不大**

```bash
get agg embedding time 0.21804475784301758
get nodes embedding time 0.002117156982421875
post llm use time:  1.2400257587432861
query time:  1.4748468399047852
```

复杂的python计算主要在于：

- **get nodes embedding（与所有node比较相似度与topk）**，这个是线性增长的损耗。文档越多，消耗越大，如我们将文档大小翻十倍，消耗时间也翻十倍，
- query中再次出现的多次token split切分（详见transformation），但这块由于选好context了，所以相对占比很小

## 5. Reference

- [llama_index_doc](https://docs.llamaindex.ai/en/stable/)
- [llama_index_github](https://github.com/run-llama/llama_index)
