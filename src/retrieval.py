from typing import List, Dict, Optional
from src.embedding_manager import EmbeddingManager
from src import config

class VariantRetriever:
    def __init__(self):
        self.manager = EmbeddingManager()
        self.manager.index = self.manager.pc.Index(config.PINECONE_INDEX_NAME)
    
    def retrieve_contexts(self, query: str, top_k: int = None, 
                         cancer_type: Optional[str] = None) -> Dict:
        if top_k is None:
            top_k = config.TOP_K
        
        # Prepare filter for hybrid search
        filter_dict = None
        if cancer_type:
            filter_dict = {"cancer_type": {"$eq": cancer_type}}
        
        # Query Pinecone
        results = self.manager.query_similar_variants(
            query_text=query,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        # Check if we have sufficient results
        if not results:
            return {
                'success': False,
                'message': 'Insufficient data. No relevant genomic variants found for your query.',
                'contexts': [],
                'query': query
            }
        
        # Format contexts for generation
        contexts = []
        for idx, result in enumerate(results, 1):
            metadata = result['metadata']
            context = {
                'rank': idx,
                'score': result['score'],
                'variant': metadata.get('variant', 'N/A'),
                'gene': metadata.get('gene', 'N/A'),
                'interpretation': metadata.get('interpretation', 'N/A'),
                'cancer_type': metadata.get('cancer_type', 'N/A'),
                'treatment': metadata.get('treatment', 'N/A'),
                'drug': metadata.get('drug', 'N/A'),
                'evidence_level': metadata.get('evidence_level', 'N/A'),
                'source': metadata.get('source', 'N/A'),
                'clinical_significance': metadata.get('clinical_significance', 'N/A')
            }
            contexts.append(context)
        
        return {
            'success': True,
            'message': f'Found {len(contexts)} relevant variants',
            'contexts': contexts,
            'query': query,
            'filter_applied': cancer_type
        }
    
    def format_contexts_for_prompt(self, contexts: List[Dict]) -> str:
        if not contexts:
            return "No relevant information found."
        
        formatted = []
        for ctx in contexts:
            context_text = f"""
Context {ctx['rank']} (Relevance: {ctx['score']:.2f}):
- Variant: {ctx['variant']}
- Gene: {ctx['gene']}
- Cancer Type: {ctx['cancer_type']}
- Clinical Significance: {ctx['clinical_significance']}
- Interpretation: {ctx['interpretation']}
- Treatment: {ctx['treatment']}
- Drug: {ctx['drug']}
- Evidence Level: {ctx['evidence_level']}
- Source: {ctx['source']}
""".strip()
            formatted.append(context_text)
        
        return "\n\n".join(formatted)
    
    def extract_citations(self, contexts: List[Dict]) -> List[str]:
        citations = []
        for ctx in contexts:
            citation = f"[{ctx['rank']}] {ctx['source']} - {ctx['variant']} in {ctx['gene']} (Evidence: {ctx['evidence_level']})"
            citations.append(citation)
        return citations

def test_retrieval():
    """Test the retrieval system"""
    retriever = VariantRetriever()
    
    # Test queries
    test_queries = [
        "Best drug for BRCA1 mutation?",
        "Treatment options for TP53 mutations in breast cancer",
        "What are the implications of EGFR mutations?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = retriever.retrieve_contexts(query)
        
        if result['success']:
            print(f"\n{result['message']}")
            print(f"\nRetrieved Contexts:")
            print(retriever.format_contexts_for_prompt(result['contexts']))
            print(f"\nCitations:")
            for citation in retriever.extract_citations(result['contexts']):
                print(f"  {citation}")
        else:
            print(f"\n{result['message']}")

if __name__ == "__main__":
    test_retrieval()
