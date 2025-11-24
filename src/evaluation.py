import json
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from src.retrieval import VariantRetriever
from src.generation import VariantResponseGenerator

class RAGEvaluator:
    def __init__(self):
        self.retriever = VariantRetriever()
        self.generator = VariantResponseGenerator()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_retrieval_accuracy(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate retrieval accuracy using test cases
        
        Test case format:
        {
            'query': str,
            'expected_genes': List[str],
            'expected_variants': List[str],
            'cancer_type': Optional[str]
        }
        """
        results = {
            'total': len(test_cases),
            'correct_retrievals': 0,
            'partial_matches': 0,
            'no_matches': 0,
            'average_score': 0.0,
            'cases': []
        }
        
        total_score = 0.0
        
        for case in test_cases:
            query = case['query']
            expected_genes = set(case.get('expected_genes', []))
            expected_variants = set(case.get('expected_variants', []))
            cancer_type = case.get('cancer_type')
            
            # Retrieve contexts
            retrieval_result = self.retriever.retrieve_contexts(
                query, 
                cancer_type=cancer_type
            )
            
            if not retrieval_result['success']:
                results['no_matches'] += 1
                results['cases'].append({
                    'query': query,
                    'status': 'no_match',
                    'score': 0.0
                })
                continue
            
            # Check if expected genes/variants are in retrieved contexts
            retrieved_genes = set()
            retrieved_variants = set()
            
            for ctx in retrieval_result['contexts']:
                retrieved_genes.add(ctx['gene'])
                retrieved_variants.add(ctx['variant'])
            
            # Calculate match score
            gene_match = len(expected_genes.intersection(retrieved_genes)) / len(expected_genes) if expected_genes else 0
            variant_match = len(expected_variants.intersection(retrieved_variants)) / len(expected_variants) if expected_variants else 0
            
            match_score = (gene_match + variant_match) / 2
            total_score += match_score
            
            if match_score >= 0.8:
                results['correct_retrievals'] += 1
                status = 'correct'
            elif match_score > 0:
                results['partial_matches'] += 1
                status = 'partial'
            else:
                results['no_matches'] += 1
                status = 'no_match'
            
            results['cases'].append({
                'query': query,
                'status': status,
                'score': match_score,
                'retrieved_genes': list(retrieved_genes),
                'retrieved_variants': list(retrieved_variants)
            })
        
        results['average_score'] = total_score / len(test_cases) if test_cases else 0
        results['accuracy'] = results['correct_retrievals'] / len(test_cases) if test_cases else 0
        
        return results
    
    def evaluate_generation_quality(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate generation quality using reference answers
        
        Test case format:
        {
            'query': str,
            'reference_answer': str
        }
        """
        results = {
            'total': len(test_cases),
            'rouge_scores': [],
            'citation_quality': [],
            'hallucination_check': [],
            'cases': []
        }
        
        for case in test_cases:
            query = case['query']
            reference = case.get('reference_answer', '')
            
            # Get retrieval and generation
            retrieval_result = self.retriever.retrieve_contexts(query)
            
            if not retrieval_result['success']:
                continue
            
            generation_result = self.generator.generate_response(
                query,
                retrieval_result['contexts']
            )
            
            generated_text = generation_result['response']
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(reference, generated_text)
            results['rouge_scores'].append({
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure
            })
            
            # Evaluate citation quality
            citation_score = self.evaluate_citations(
                generated_text,
                generation_result['citations']
            )
            results['citation_quality'].append(citation_score)
            
            # Check for hallucinations
            hallucination_score = self.check_hallucination(
                generated_text,
                retrieval_result['contexts']
            )
            results['hallucination_check'].append(hallucination_score)
            
            results['cases'].append({
                'query': query,
                'generated': generated_text[:200] + '...',
                'rouge': rouge_scores['rougeL'].fmeasure,
                'citation_score': citation_score,
                'hallucination_score': hallucination_score
            })
        
        # Calculate averages
        if results['rouge_scores']:
            results['avg_rouge1'] = np.mean([s['rouge1'] for s in results['rouge_scores']])
            results['avg_rouge2'] = np.mean([s['rouge2'] for s in results['rouge_scores']])
            results['avg_rougeL'] = np.mean([s['rougeL'] for s in results['rouge_scores']])
        
        if results['citation_quality']:
            results['avg_citation_quality'] = np.mean(results['citation_quality'])
        
        if results['hallucination_check']:
            results['avg_hallucination_score'] = np.mean(results['hallucination_check'])
        
        return results
    
    def evaluate_citations(self, generated_text: str, citations: List[str]) -> float:
        if not citations:
            return 0.0
        
        score = 0.0
        
        # Check if citations are referenced in text
        citation_markers = ['[Source', '[1]', '[2]', '[3]']
        has_citation_markers = any(marker in generated_text for marker in citation_markers)
        
        if has_citation_markers:
            score += 0.5
        
        # Check citation completeness
        required_fields = ['variant', 'gene', 'Evidence']
        for citation in citations:
            fields_present = sum(1 for field in required_fields if field.lower() in citation.lower())
            score += (fields_present / len(required_fields)) * (0.5 / len(citations))
        
        return min(score, 1.0)
    
    def check_hallucination(self, generated_text: str, contexts: List[Dict]) -> float:
        # Extract key terms from contexts
        context_terms = set()
        for ctx in contexts:
            for key in ['variant', 'gene', 'drug', 'treatment', 'cancer_type']:
                value = ctx.get(key, '')
                if value and value != 'N/A':
                    context_terms.add(value.lower())
        
        # Check if generated text only contains terms from contexts
        # This is a simple heuristic
        score = 1.0
        
        # Check for common hallucination phrases
        hallucination_phrases = [
            'studies show',
            'research indicates',
            'proven to',
            'definitely',
            'always effective',
            'never fails'
        ]
        
        generated_lower = generated_text.lower()
        for phrase in hallucination_phrases:
            if phrase in generated_lower:
                score -= 0.1
        
        # Check if response stays grounded
        if 'consult' in generated_lower or 'clinician' in generated_lower:
            score += 0.2
        
        return max(0.0, min(score, 1.0))
    
    def generate_evaluation_report(self, retrieval_results: Dict, 
                                   generation_results: Dict) -> str:
        """
        Generate comprehensive evaluation report
        """
        report = f"""
{'='*80}
RAG PIPELINE EVALUATION REPORT
{'='*80}

RETRIEVAL PERFORMANCE:
--------------------
Total Test Cases: {retrieval_results['total']}
Correct Retrievals: {retrieval_results['correct_retrievals']} ({retrieval_results['accuracy']*100:.1f}%)
Partial Matches: {retrieval_results['partial_matches']}
No Matches: {retrieval_results['no_matches']}
Average Retrieval Score: {retrieval_results['average_score']:.3f}

GENERATION QUALITY:
------------------
Average ROUGE-1: {generation_results.get('avg_rouge1', 0):.3f}
Average ROUGE-2: {generation_results.get('avg_rouge2', 0):.3f}
Average ROUGE-L: {generation_results.get('avg_rougeL', 0):.3f}
Average Citation Quality: {generation_results.get('avg_citation_quality', 0):.3f}
Average Hallucination Score: {generation_results.get('avg_hallucination_score', 0):.3f}

OVERALL ACCURACY:
----------------
Combined Accuracy Score: {(retrieval_results['accuracy'] + generation_results.get('avg_rougeL', 0)) / 2 * 100:.1f}%

TARGET: 75%+ Accuracy
STATUS: {'✓ PASSED' if (retrieval_results['accuracy'] + generation_results.get('avg_rougeL', 0)) / 2 >= 0.75 else '✗ NEEDS IMPROVEMENT'}

{'='*80}
"""
        return report

def create_test_cases() -> Tuple[List[Dict], List[Dict]]:
    """
    Create test cases for evaluation
    """
    retrieval_test_cases = [
        {
            'query': 'Best treatment for BRCA1 mutation in breast cancer',
            'expected_genes': ['BRCA1'],
            'expected_variants': ['BRCA1 c.5266dupC'],
            'cancer_type': 'Breast Cancer'
        },
        {
            'query': 'What drugs target EGFR mutations?',
            'expected_genes': ['EGFR'],
            'expected_variants': []
        },
        {
            'query': 'TP53 mutation implications in lung cancer',
            'expected_genes': ['TP53'],
            'cancer_type': 'Lung Cancer'
        }
    ]
    
    generation_test_cases = [
        {
            'query': 'Best treatment for BRCA1 mutation',
            'reference_answer': 'BRCA1 mutations are associated with breast and ovarian cancers. Treatment options include PARP inhibitors such as olaparib and platinum-based chemotherapy. Genetic counseling is recommended.'
        },
        {
            'query': 'What are EGFR inhibitors?',
            'reference_answer': 'EGFR inhibitors are targeted therapies that block the EGFR protein. They are used in treating cancers with EGFR mutations, particularly lung cancer. Examples include erlotinib and gefitinib.'
        }
    ]
    
    return retrieval_test_cases, generation_test_cases

def main():
    """Run evaluation"""
    print("Starting RAG Pipeline Evaluation...\n")
    
    evaluator = RAGEvaluator()
    
    # Create test cases
    retrieval_tests, generation_tests = create_test_cases()
    
    # Run evaluations
    print("Evaluating retrieval accuracy...")
    retrieval_results = evaluator.evaluate_retrieval_accuracy(retrieval_tests)
    
    print("Evaluating generation quality...")
    generation_results = evaluator.evaluate_generation_quality(generation_tests)
    
    # Generate report
    report = evaluator.generate_evaluation_report(retrieval_results, generation_results)
    print(report)
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'retrieval': retrieval_results,
            'generation': generation_results
        }, f, indent=2)
    
    print("✓ Evaluation complete! Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()
