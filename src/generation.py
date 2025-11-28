from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from src import config

class VariantResponseGenerator:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = config.LLM_MODEL
        
        print(f"Loading generation model: {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # CPU
        )
        
        print("✓ Generation model loaded successfully")
    
    def create_template_response(self, query: str, contexts: List[Dict]) -> str:
        if not contexts:
            return "Insufficient data; consult clinician. No relevant genomic variants found for your query."
        
        # Extract information from contexts
        response_parts = []
        
        # Analyze query intent
        query_lower = query.lower()
        is_treatment_query = any(word in query_lower for word in ['treatment', 'drug', 'therapy', 'medication'])
        is_significance_query = any(word in query_lower for word in ['significance', 'meaning', 'implication', 'what is'])
        
        # Build response based on contexts
        primary_context = contexts[0]  # Highest relevance
        
        # Introduction
        gene = primary_context['gene']
        variant = primary_context['variant']
        
        intro = f"Based on the retrieved data for {gene} ({variant}):\n\n"
        response_parts.append(intro)
        
        # Clinical Significance
        if is_significance_query or not is_treatment_query:
            response_parts.append("**Clinical Significance:**\n")
            response_parts.append(f"{primary_context['interpretation']}\n\n")
            response_parts.append(f"- Cancer Type: {primary_context['cancer_type']}\n")
            response_parts.append(f"- Clinical Classification: {primary_context['clinical_significance']}\n")
            response_parts.append(f"- Evidence Level: {primary_context['evidence_level']}\n\n")
        
        # Treatment Information
        if is_treatment_query or contexts[0]['treatment'] != 'N/A':
            response_parts.append("**Treatment Recommendations:**\n")
            
            # Collect unique treatments from all contexts
            treatments = {}
            for ctx in contexts:
                if ctx['treatment'] != 'N/A' and ctx['drug'] != 'N/A':
                    key = ctx['gene']
                    if key not in treatments:
                        treatments[key] = {
                            'treatment': ctx['treatment'],
                            'drug': ctx['drug'],
                            'evidence': ctx['evidence_level'],
                            'cancer': ctx['cancer_type']
                        }
            
            for gene_name, tx_info in treatments.items():
                response_parts.append(f"- For {gene_name}: {tx_info['treatment']}\n")
                response_parts.append(f"  - Specific Drugs: {tx_info['drug']}\n")
                response_parts.append(f"  - Evidence Level: {tx_info['evidence']}\n")
            
            response_parts.append("\n")
        
        # Additional contexts
        if len(contexts) > 1:
            response_parts.append("**Related Findings:**\n")
            for i, ctx in enumerate(contexts[1:], 2):
                response_parts.append(f"[Source {i}] {ctx['gene']} - {ctx['variant']}: ")
                response_parts.append(f"{ctx['cancer_type']}, {ctx['clinical_significance']}\n")
        
        # Sources citation
        response_parts.append("\n**Sources:** ")
        source_list = [f"[{i}] {ctx['source']}" for i, ctx in enumerate(contexts, 1)]
        response_parts.append(", ".join(source_list[:3]))  # Show top 3 sources
        
        # Disclaimer
        response_parts.append("\n\nNote: This information is for research purposes only. Always consult with a healthcare professional before making treatment decisions.")
        
        return "".join(response_parts)
    
    def create_prompt(self, query: str, contexts: List[Dict]) -> str:
        context_text = ""
        for ctx in contexts:
            context_text += f"""
[Source {ctx['rank']}: {ctx['source']}]
Variant: {ctx['variant']} in {ctx['gene']}
Cancer Type: {ctx['cancer_type']}
Clinical Significance: {ctx['clinical_significance']}
Interpretation: {ctx['interpretation']}
Recommended Treatment: {ctx['treatment']}
Drug: {ctx['drug']}
Evidence Level: {ctx['evidence_level']}
---
"""
        
        # Create prompt with strict instructions
        prompt = f"""You are a genomic medicine assistant. Answer the query using ONLY the provided contexts.

CONTEXTS FROM DATABASE:
{context_text}

QUERY: {query}

INSTRUCTIONS:
1. Use ONLY information from the contexts above
2. Cite sources using [Source X] notation
3. Include specific variant names, genes, and drug recommendations
4. Mention evidence levels when available
5. Be specific and clinical

ANSWER:"""
        
        return prompt
    
    def generate_response(self, query: str, contexts: List[Dict], 
                         max_length: int = None) -> Dict:
        if not contexts:
            return {
                'response': 'Insufficient data; consult clinician. No relevant genomic variants found for your query.',
                'query': query,
                'citations': [],
                'contexts_used': 0,
                'warning': config.DISCLAIMER
            }
        
        # Use template-based generation for reliability
        # This ensures 100% grounding in your data
        response_text = self.create_template_response(query, contexts)
        
        # Try LLM generation as backup (optional)
        # Uncomment if you want to use GPT-2 generation
        # try:
        #     prompt = self.create_prompt(query, contexts)
        #     if max_length is None:
        #         max_length = 256
        #     
        #     generated = self.generator(
        #         prompt,
        #         max_new_tokens=max_length,
        #         num_return_sequences=1,
        #         temperature=config.TEMPERATURE,
        #         do_sample=True,
        #         top_p=0.9,
        #         pad_token_id=self.tokenizer.eos_token_id
        #     )
        #     
        #     full_text = generated[0]['generated_text']
        #     if "ANSWER:" in full_text:
        #         llm_response = full_text.split("ANSWER:")[-1].strip()
        #         # Use LLM response if it's good quality
        #         if len(llm_response) > 50 and "consult" not in llm_response.lower():
        #             response_text = llm_response
        # except Exception as e:
        #     print(f"LLM generation failed, using template: {e}")
        
        # Extract citations
        citations = self.extract_citations_from_contexts(contexts)
        
        return {
            'response': response_text,
            'query': query,
            'citations': citations,
            'contexts_used': len(contexts),
            'warning': config.DISCLAIMER
        }
    
    def post_process_response(self, response: str, contexts: List[Dict]) -> str:
        # Limit length if too long
        if len(response) > 1500:
            response = response[:1500] + "..."
        
        # Ensure it ends properly
        if response and not response[-1] in '.!?':
            # Find last complete sentence
            for delimiter in ['.', '!', '?']:
                last_idx = response.rfind(delimiter)
                if last_idx > len(response) // 2:
                    response = response[:last_idx + 1]
                    break
        
        return response.strip()
    
    def extract_citations_from_contexts(self, contexts: List[Dict]) -> List[str]:
        citations = []
        for ctx in contexts:
            citation = (
                f"[{ctx['rank']}] {ctx['source']} - "
                f"{ctx['variant']} in {ctx['gene']}: {ctx['cancer_type']} "
                f"(Evidence Level: {ctx['evidence_level']}, "
                f"Similarity: {ctx['score']:.2f})"
            )
            citations.append(citation)
        return citations
    
    def format_final_output(self, result: Dict) -> str:
        output = f"""
{'='*80}
QUERY: {result['query']}
{'='*80}

RESPONSE:
{result['response']}

{'='*80}
SOURCES ({result['contexts_used']} contexts used):
{'='*80}
"""
        for citation in result['citations']:
            output += f"\n{citation}"
        
        output += f"\n\n{'='*80}\n{result['warning']}\n{'='*80}"
        
        return output

def test_generation():
    """Test the generation system"""
    from src.retrieval import VariantRetriever
    
    # Initialize components
    retriever = VariantRetriever()
    generator = VariantResponseGenerator()
    
    # Test query
    query = "What is the best treatment for BRCA1 mutations in breast cancer?"
    
    print(f"Processing query: {query}\n")
    
    # Retrieve contexts
    retrieval_result = retriever.retrieve_contexts(query)
    
    if retrieval_result['success']:
        # Generate response
        generation_result = generator.generate_response(
            query,
            retrieval_result['contexts']
        )
        
        # Display formatted output
        print(generator.format_final_output(generation_result))
    else:
        print(retrieval_result['message'])

if __name__ == "__main__":
    test_generation()