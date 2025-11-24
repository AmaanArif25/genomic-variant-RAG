import sys
import argparse
from src.embedding_manager import EmbeddingManager
from src.retrieval import VariantRetriever
from src.generation import VariantResponseGenerator
from src.evaluation import RAGEvaluator, create_test_cases
from src import config

def setup_pipeline():
    print("="*80)
    print("GENOMIC RAG PIPELINE - SETUP")
    print("="*80)
    
    # Step 1: Initialize embedding manager
    print("\n[1/3] Initializing Embedding Manager...")
    manager = EmbeddingManager()
    
    # Step 2: Create Pinecone index
    print("\n[2/3] Creating Pinecone Index...")
    manager.create_index()
    
    # Step 3: Load and upsert variants
    print("\n[3/3] Loading and Indexing Variants...")
    variants = manager.load_variants(config.DATA_PATH)
    manager.upsert_to_pinecone(variants)
    
    print("\n" + "="*80)
    print("✓ SETUP COMPLETE")
    print("="*80)
    print(f"Total variants indexed: {len(variants)}")
    print(f"Index name: {config.PINECONE_INDEX_NAME}")
    print(f"Embedding dimension: {config.EMBEDDING_DIM}")

def test_query(query: str, cancer_type: str = None):
    """
    Test a single query through the complete RAG pipeline
    """
    print("\n" + "="*80)
    print("TESTING RAG PIPELINE")
    print("="*80)
    
    # Initialize components
    print("\nInitializing components...")
    retriever = VariantRetriever()
    generator = VariantResponseGenerator()
    
    # Retrieval
    print(f"\nQuery: {query}")
    if cancer_type:
        print(f"Cancer Type Filter: {cancer_type}")
    
    print("\n[1/2] Retrieving relevant contexts...")
    retrieval_result = retriever.retrieve_contexts(query, cancer_type=cancer_type)
    
    if not retrieval_result['success']:
        print(f"\n✗ {retrieval_result['message']}")
        return
    
    print(f"✓ Found {len(retrieval_result['contexts'])} relevant variants")
    
    # Display contexts
    print("\nRetrieved Contexts:")
    print("-" * 80)
    for ctx in retrieval_result['contexts']:
        print(f"\n[{ctx['rank']}] {ctx['variant']} in {ctx['gene']} (Score: {ctx['score']:.3f})")
        print(f"    Cancer Type: {ctx['cancer_type']}")
        print(f"    Treatment: {ctx['treatment']}")
        print(f"    Drug: {ctx['drug']}")
    
    # Generation
    print("\n[2/2] Generating response...")
    generation_result = generator.generate_response(
        query,
        retrieval_result['contexts']
    )
    
    # Display result
    print("\n" + "="*80)
    print("FINAL RESPONSE")
    print("="*80)
    print(f"\n{generation_result['response']}")
    
    print("\n" + "="*80)
    print(f"CITATIONS ({generation_result['contexts_used']} sources)")
    print("="*80)
    for citation in generation_result['citations']:
        print(f"\n{citation}")
    
    print("\n" + "="*80)
    print(generation_result['warning'])
    print("="*80)

def run_evaluation():
    """
    Run complete evaluation suite
    """
    print("\n" + "="*80)
    print("RUNNING EVALUATION SUITE")
    print("="*80)
    
    evaluator = RAGEvaluator()
    
    # Create test cases
    print("\nCreating test cases...")
    retrieval_tests, generation_tests = create_test_cases()
    
    # Run evaluations
    print("\n[1/2] Evaluating retrieval accuracy...")
    retrieval_results = evaluator.evaluate_retrieval_accuracy(retrieval_tests)
    
    print("\n[2/2] Evaluating generation quality...")
    generation_results = evaluator.evaluate_generation_quality(generation_tests)
    
    # Generate report
    report = evaluator.generate_evaluation_report(retrieval_results, generation_results)
    print("\n" + report)
    
    # Save results
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'retrieval': retrieval_results,
            'generation': generation_results
        }, f, indent=2)
    
    print("\n✓ Results saved to evaluation_results.json")

def interactive_mode():
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nType 'exit' to quit, 'help' for commands\n")
    
    # Initialize components
    retriever = VariantRetriever()
    generator = VariantResponseGenerator()
    
    while True:
        try:
            query = input("\n🧬 Enter query: ").strip()
            
            if query.lower() == 'exit':
                print("\nGoodbye!")
                break
            
            if query.lower() == 'help':
                print("\nCommands:")
                print("  - Type any genomic query")
                print("  - 'exit' to quit")
                print("  - 'help' for this message")
                continue
            
            if not query:
                continue
            
            # Retrieve
            retrieval_result = retriever.retrieve_contexts(query)
            
            if not retrieval_result['success']:
                print(f"\n✗ {retrieval_result['message']}")
                continue
            
            # Generate
            generation_result = generator.generate_response(
                query,
                retrieval_result['contexts']
            )
            
            # Display
            print("\n" + "="*80)
            print("RESPONSE:")
            print("="*80)
            print(f"\n{generation_result['response']}")
            
            print("\n" + "="*80)
            print(f"RETRIEVED CONTEXTS ({len(retrieval_result['contexts'])} variants from YOUR data):")
            print("="*80)
            for ctx in retrieval_result['contexts']:
                print(f"\n[{ctx['rank']}] {ctx['variant']} ({ctx['gene']}) - {ctx['cancer_type']}")
                print(f"    Score: {ctx['score']:.3f} | Evidence: {ctx['evidence_level']}")
                print(f"    Treatment: {ctx['treatment']}")
                print(f"    Drug: {ctx['drug']}")
            
            print("\n" + "="*80)
            print("CITATIONS:")
            print("="*80)
            for i, citation in enumerate(generation_result['citations'], 1):
                print(f"{i}. {citation}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Genomic Variant RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Complete RAG pipeline for genomic variant interpretation using Pinecone.
        """
    )
    
    parser.add_argument('--setup', action='store_true', 
                       help='Setup pipeline: create index and ingest data')
    parser.add_argument('--query', type=str, 
                       help='Test a single query')
    parser.add_argument('--cancer-type', type=str, 
                       help='Filter by cancer type (use with --query)')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Run complete evaluation suite')
    parser.add_argument('--interactive', action='store_true', 
                       help='Start interactive query mode')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Execute based on arguments
    if args.setup:
        setup_pipeline()
    
    if args.query:
        test_query(args.query, args.cancer_type)
    
    if args.evaluate:
        run_evaluation()
    
    if args.interactive:
        interactive_mode()

if __name__ == "__main__":
    main()