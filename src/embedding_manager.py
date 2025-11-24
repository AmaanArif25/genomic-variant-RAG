import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from src import config

class EmbeddingManager:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        print("Initializing Pinecone...")
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = None
        
    def create_index(self):
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if config.PINECONE_INDEX_NAME not in existing_indexes:
                print(f"Creating index: {config.PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=config.PINECONE_INDEX_NAME,
                    dimension=config.EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            else:
                print(f"Index {config.PINECONE_INDEX_NAME} already exists")
            
            self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
            print("Index ready!")
            
        except Exception as e:
            print(f"Error creating index: {e}")
            raise
    
    def load_variants(self, filepath: str) -> List[Dict]:
        print(f"Loading variants from {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict) and 'variants' in data:
            variants = data['variants']
        elif isinstance(data, list):
            variants = data
        else:
            raise ValueError("Unexpected JSON format")
        
        print(f"Loaded {len(variants)} variants")
        return variants
    
    def prepare_text_for_embedding(self, variant: Dict) -> str:
        parts = []
        
        # Core variant info
        if 'variant' in variant:
            parts.append(f"Variant: {variant['variant']}")
        if 'gene' in variant:
            parts.append(f"Gene: {variant['gene']}")
        
        # Clinical interpretation
        if 'interpretation' in variant:
            parts.append(f"Interpretation: {variant['interpretation']}")
        
        # Cancer type and clinical significance
        if 'cancer_type' in variant:
            parts.append(f"Cancer Type: {variant['cancer_type']}")
        if 'clinical_significance' in variant:
            parts.append(f"Clinical Significance: {variant['clinical_significance']}")
        
        # Treatment information
        if 'treatment' in variant:
            parts.append(f"Treatment: {variant['treatment']}")
        if 'drug' in variant:
            parts.append(f"Drug: {variant['drug']}")
        
        # Evidence and source
        if 'evidence_level' in variant:
            parts.append(f"Evidence Level: {variant['evidence_level']}")
        if 'source' in variant:
            parts.append(f"Source: {variant['source']}")
        
        return " | ".join(parts)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        return self.model.encode(texts, show_progress_bar=True).tolist()
    
    def upsert_to_pinecone(self, variants: List[Dict], batch_size: int = 100):
        """
        Upsert variant embeddings to Pinecone with metadata
        """
        if not self.index:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        print(f"Processing {len(variants)} variants for upsert...")
        
        vectors_to_upsert = []
        
        for idx, variant in enumerate(tqdm(variants, desc="Preparing embeddings")):
            # Prepare text for embedding
            text = self.prepare_text_for_embedding(variant)
            
            # Generate embedding
            embedding = self.model.encode([text])[0].tolist()
            
            # Prepare metadata (ensure all values are JSON-serializable)
            metadata = {
                'variant': str(variant.get('variant', '')),
                'gene': str(variant.get('gene', '')),
                'interpretation': str(variant.get('interpretation', ''))[:1000],  # Truncate long texts
                'cancer_type': str(variant.get('cancer_type', '')),
                'clinical_significance': str(variant.get('clinical_significance', '')),
                'treatment': str(variant.get('treatment', '')),
                'drug': str(variant.get('drug', '')),
                'evidence_level': str(variant.get('evidence_level', '')),
                'source': str(variant.get('source', '')),
                'text': text[:1000]  # Store searchable text
            }
            
            # Create vector tuple (id, embedding, metadata)
            vector_id = f"variant_{idx}"
            vectors_to_upsert.append((vector_id, embedding, metadata))
            
            # Batch upsert
            if len(vectors_to_upsert) >= batch_size:
                self.index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
        
        # Upsert remaining vectors
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
        
        print(f"✓ Successfully upserted {len(variants)} variants to Pinecone")
        
        # Get index stats
        stats = self.index.describe_index_stats()
        print(f"Index stats: {stats}")
    
    def query_similar_variants(self, query_text: str, top_k: int = 3, 
                              filter_dict: Dict = None) -> List[Dict]:
        """
        Query Pinecone for similar variants
        """
        if not self.index:
            raise ValueError("Index not initialized")
        
        # Generate query embedding
        query_embedding = self.model.encode([query_text])[0].tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            if match['score'] >= config.SIMILARITY_THRESHOLD:
                formatted_results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match['metadata']
                })
        
        return formatted_results

def main():
    """Main function for data ingestion"""
    # Initialize manager
    manager = EmbeddingManager()
    
    # Create index
    manager.create_index()
    
    # Load variants
    variants = manager.load_variants(config.DATA_PATH)
    
    # Show what will be indexed
    print("\n" + "="*80)
    print("VARIANTS TO BE INDEXED (from variants.json):")
    print("="*80)
    for i, v in enumerate(variants, 1):
        print(f"{i}. {v.get('gene', 'N/A')} - {v.get('variant', 'N/A')} ({v.get('cancer_type', 'N/A')})")
    print("="*80)
    
    # Upsert to Pinecone
    manager.upsert_to_pinecone(variants)
    
    print("\n✓ Data ingestion complete!")
    print(f"Total variants indexed: {len(variants)}")
    print(f"Source file: {config.DATA_PATH}")
    print("\n🔍 All queries will ONLY return results from these {len(variants)} variants!")

if __name__ == "__main__":
    main()