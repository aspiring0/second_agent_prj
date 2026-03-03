//! RAG系统核心模块 - Rust测试文件
//! 用于测试Rust代码的切分效果

use std::collections::HashMap;

/// 文档结构体
#[derive(Clone, Debug)]
pub struct Document {
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub source: String,
}

impl Document {
    pub fn new(content: String, source: String) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), source.clone());
        Document {
            content,
            metadata,
            source,
        }
    }
}

/// 向量存储
pub struct VectorStore {
    persist_dir: String,
    embedding_model: String,
    index: Vec<Document>,
    embeddings: Vec<Vec<f32>>,
}

impl VectorStore {
    /// 创建向量存储
    pub fn new(persist_dir: &str, embedding_model: &str) -> Self {
        VectorStore {
            persist_dir: persist_dir.to_string(),
            embedding_model: embedding_model.to_string(),
            index: Vec::new(),
            embeddings: Vec::new(),
        }
    }

    /// 添加文档到向量存储
    pub fn add_documents(&mut self, documents: &[Document]) -> usize {
        let mut count = 0;
        for doc in documents {
            let embedding = self.generate_embedding(&doc.content);
            self.index.push(doc.clone());
            self.embeddings.push(embedding);
            count += 1;
        }
        count
    }

    /// 搜索相似文档
    pub fn search(&self, query: &str, top_k: usize) -> Vec<&Document> {
        let query_embedding = self.generate_embedding(query);

        // 计算相似度并收集结果
        let mut results: Vec<(usize, f32)> = self
            .index
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let score = self.cosine_similarity(&query_embedding, &self.embeddings[i]);
                (i, score)
            })
            .collect();

        // 按相似度排序
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 返回top_k个文档
        results
            .iter()
            .take(top_k)
            .map(|(i, _)| &self.index[*i])
            .collect()
    }

    /// 生成文本向量
    fn generate_embedding(&self, _text: &str) -> Vec<f32> {
        // 调用Embedding API（简化实现）
        vec![0.0; 1536]
    }

    /// 计算余弦相似度
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot_product / (norm_a * norm_b)
    }
}

/// RAG流水线
pub struct RAGPipeline {
    vector_store: VectorStore,
    llm_model: String,
}

impl RAGPipeline {
    /// 创建RAG流水线
    pub fn new(vector_store: VectorStore, llm_model: &str) -> Self {
        RAGPipeline {
            vector_store,
            llm_model: llm_model.to_string(),
        }
    }

    /// 执行RAG查询
    pub fn query(&self, question: &str) -> String {
        // 1. 检索
        let docs = self.vector_store.search(question, 5);

        // 2. 构建上下文
        let context: String = docs
            .iter()
            .map(|d| d.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        // 3. 生成回答
        let prompt = self.build_prompt(question, &context);
        self.call_llm(&prompt)
    }

    /// 构建提示词
    fn build_prompt(&self, question: &str, context: &str) -> String {
        format!(
            r#"基于以下上下文回答问题：

上下文：
{}

问题：{}"#,
            context, question
        )
    }

    /// 调用LLM
    fn call_llm(&self, _prompt: &str) -> String {
        // 调用LLM API（简化实现）
        "这是一个示例回答".to_string()
    }
}

/// ETL处理器
pub struct ETLProcessor {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl ETLProcessor {
    /// 创建ETL处理器
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        ETLProcessor {
            chunk_size,
            chunk_overlap,
        }
    }

    /// 切分文档
    pub fn split_document(&self, document: &Document) -> Vec<Document> {
        let mut chunks = Vec::new();
        let content = document.content.as_bytes();
        let mut start = 0;

        while start < content.len() {
            let end = std::cmp::min(start + self.chunk_size, content.len());
            let chunk = String::from_utf8_lossy(&content[start..end]).to_string();

            let mut metadata = document.metadata.clone();
            metadata.insert(
                "chunk_index".to_string(),
                (start / self.chunk_size).to_string(),
            );

            chunks.push(Document {
                content: chunk,
                metadata,
                source: document.source.clone(),
            });

            start += self.chunk_size;
            if start >= content.len() {
                break;
            }
            start = start.saturating_sub(self.chunk_overlap);
        }

        chunks
    }
}

/// 主函数
fn main() {
    let mut vs = VectorStore::new("./data", "text-embedding-3-small");
    
    // 添加测试文档
    let doc = Document::new(
        "RAG是检索增强生成技术".to_string(),
        "test.txt".to_string(),
    );
    vs.add_documents(&[doc]);

    let pipeline = RAGPipeline::new(vs, "gpt-4");
    let answer = pipeline.query("什么是RAG？");
    println!("{}", answer);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_store() {
        let mut vs = VectorStore::new("./test_data", "test-model");
        let doc = Document::new("测试内容".to_string(), "test.txt".to_string());
        
        let count = vs.add_documents(&[doc]);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_etl_processor() {
        let processor = ETLProcessor::new(100, 20);
        let doc = Document::new("这是一个测试文档".repeat(20), "test.txt".to_string());
        
        let chunks = processor.split_document(&doc);
        assert!(!chunks.is_empty());
    }
}