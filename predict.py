import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os

class VariantPredictor:
    def __init__(self, model_dir='./biobert_variant_model'):        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}\nPlease train the model first using: python train_biobert.py")
        
        print(f"Loading model from {model_dir}...")
        
        # Load model and tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label encoder
        label_encoder_path = f'{model_dir}/label_encoder.pkl'
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")
        
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}\n")
    
    def predict(self, query, return_probabilities=False):
        # Tokenize input
        inputs = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred_idx = torch.argmax(logits, dim=-1).item()
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        
        predicted_label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        
        result = {
            'query': query,
            'prediction': predicted_label,
            'confidence': confidence
        }
        
        if return_probabilities:
            # Get top 3 predictions
            top_indices = probs.argsort()[-3:][::-1]
            top_predictions = [
                {
                    'label': self.label_encoder.inverse_transform([idx])[0],
                    'probability': float(probs[idx])
                }
                for idx in top_indices
            ]
            result['top_predictions'] = top_predictions
        
        return result
    
    def batch_predict(self, queries):
        results = []
        for query in queries:
            result = self.predict(query)
            results.append(result)
        return results

def main():
    parser = argparse.ArgumentParser(
        description='Make predictions with trained BioBERT variant classifier'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./biobert_variant_model',
        help='Directory containing the trained model'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to predict (e.g., "Query: EGFR L858R in lung?")'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--show_probs',
        action='store_true',
        help='Show probability distribution'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = VariantPredictor(args.model_dir)
        
        if args.interactive:
            # Interactive mode
            print("=" * 60)
            print("🔮 Interactive Prediction Mode")
            print("=" * 60)
            print("Enter queries (or 'quit' to exit)")
            print("Example: EGFR L858R in lung")
            print()
            
            while True:
                query = input("Enter query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Add "Query:" prefix if not present
                if not query.startswith("Query:"):
                    query = f"Query: {query}"
                
                result = predictor.predict(query, return_probabilities=args.show_probs)
                
                print("\n" + "-" * 60)
                print(f"Query: {result['query']}")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
                
                if args.show_probs and 'top_predictions' in result:
                    print("\nTop 3 Predictions:")
                    for i, pred in enumerate(result['top_predictions'], 1):
                        print(f"  {i}. {pred['label']}: {pred['probability']:.4f}")
                
                print("-" * 60 + "\n")
        
        elif args.query:
            # Single query mode
            result = predictor.predict(args.query, return_probabilities=args.show_probs)
            
            print("\n" + "=" * 60)
            print("Prediction Result")
            print("=" * 60)
            print(f"Query: {result['query']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            
            if args.show_probs and 'top_predictions' in result:
                print("\nTop 3 Predictions:")
                for i, pred in enumerate(result['top_predictions'], 1):
                    print(f"  {i}. {pred['label']}: {pred['probability']:.4f}")
            
            print("=" * 60 + "\n")
        
        else:
            # Demo mode
            print("=" * 60)
            print("Demo Predictions")
            print("=" * 60 + "\n")
            
            demo_queries = [
                "Query: TP53 p.R248W in breast?",
                "Query: EGFR L858R in lung?",
                "Query: BRCA1 c.5266dupC in ovarian?",
                "Query: KRAS G12D in colorectal?",
                "Query: PIK3CA E545K in breast?"
            ]
            
            results = predictor.batch_predict(demo_queries)
            
            for result in results:
                print(f"Query: {result['query']}")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f}\n")
            
            print("=" * 60)
    
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}\n")
    except Exception as e:
        print(f"\nError: {str(e)}\n")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
