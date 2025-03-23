import logging
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from datetime import datetime
import optuna
import random
from collections import deque
import requests
from bs4 import BeautifulSoup
import time
import threading
import numpy as np
from dataclasses import dataclass, field

# --- Configuration ---
@dataclass
class NexusConfig:
    # Model configuration
    base_model_name: str = "google/flan-t5-xxl"
    
    # Learning parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    
    # Data paths
    feedback_data_path: str = "nexus_llm_feedback.jsonl"
    model_save_path: str = "nexus_llm_model"
    performance_log_path: str = "nexus_llm_performance.log"
    model_pool_path: str = "nexus_model_pool"
    
    # Training parameters
    evaluation_interval: int = 50
    batch_size: int = 4
    max_input_length: int = 1536
    train_test_split: float = 0.05
    
    # Optimization parameters
    self_optimization_frequency: int = 100
    hyperopt_max_evals: int = 15
    optimization_metric: str = "val_loss"
    min_feedback_for_retrain: int = 50
    performance_history_length: int = 30
    
    # Knowledge retrieval
    knowledge_apis: Dict[str, str] = field(default_factory=lambda: {
        "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/DOI:",
        "wikipedia": "https://en.wikipedia.org/w/api.php"
    })
    knowledge_update_frequency: int = 3600
    
    # Hardware settings
    n_gpu: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generation parameters
    default_max_length: int = 300
    default_temperature: float = 0.7
    default_top_p: float = 0.85
    default_top_k: int = 50
    
    def __post_init__(self):
        os.makedirs(self.model_pool_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.performance_log_path), exist_ok=True)

# --- Logging Module ---
class LoggingManager:
    def __init__(self, config: NexusConfig):
        self.config = config
        
        # Set up main logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('nexus_llm')
        
        # Set up performance logger
        self.performance_logger = logging.getLogger('performance')
        self.performance_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(config.performance_log_path)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.performance_logger.addHandler(file_handler)
        
    def info(self, message: str):
        self.logger.info(message)
        
    def error(self, message: str):
        self.logger.error(message)
        
    def warning(self, message: str):
        self.logger.warning(message)
        
    def log_performance(self, message: str):
        self.performance_logger.info(message)

# --- Dataset Module ---
class FeedbackDataset(Dataset):
    def __init__(self, prompts: List[str], responses: List[str], tokenizer, max_length: int, device: torch.device):
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        full_text = self.prompts[idx] + self.tokenizer.sep_token + self.responses[idx]
        inputs = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze().to(self.device),
            'attention_mask': inputs['attention_mask'].squeeze().to(self.device),
            'labels': inputs['input_ids'].squeeze().to(self.device)
        }

# --- Knowledge Retrieval Module ---
class KnowledgeRetriever:
    def __init__(self, config: NexusConfig, logger: LoggingManager):
        self.config = config
        self.logger = logger
        self.last_update = time.time()
        self.cache = {}
        
    def needs_update(self) -> bool:
        return (time.time() - self.last_update) > self.config.knowledge_update_frequency
        
    def update_last_time(self):
        self.last_update = time.time()
        
    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract relevant keywords from text for knowledge retrieval"""
        # Simple implementation - in production would use NLP techniques
        words = text.lower().split()
        # Filter out common stopwords
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "about", "like"}
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        return keywords[:num_keywords]
        
    def retrieve_from_semantic_scholar(self, query: str) -> Optional[str]:
        """Retrieve knowledge from Semantic Scholar API"""
        if query in self.cache:
            return self.cache[query]
            
        try:
            response = requests.get(self.config.knowledge_apis["semantic_scholar"] + query, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data and 'abstract' in data:
                result = data['abstract']
            elif data and 'title' in data and 'authors' in data:
                authors = ", ".join([author['name'] for author in data['authors']])
                result = f"Title: {data['title']}, Authors: {authors}"
            else:
                return None
                
            self.cache[query] = result
            return result
            
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            self.logger.warning(f"Error during Semantic Scholar knowledge retrieval: {e}")
            return None
            
    def retrieve_from_wikipedia(self, query: str) -> Optional[str]:
        """Retrieve knowledge from Wikipedia API"""
        if query in self.cache:
            return self.cache[query]
            
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'titles': query,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
            }
            
            response = requests.get(self.config.knowledge_apis["wikipedia"], params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            if not pages:
                return None
                
            # Get the first page content
            for page_id in pages:
                page = pages[page_id]
                if 'extract' in page and page['extract']:
                    result = page['extract'][:500]  # Limit to 500 chars
                    self.cache[query] = result
                    return result
                    
            return None
            
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            self.logger.warning(f"Error during Wikipedia knowledge retrieval: {e}")
            return None
    
    def retrieve(self, prompt: str) -> Optional[str]:
        """Main method to retrieve knowledge from available sources"""
        if not self.needs_update():
            return None
            
        keywords = self.extract_keywords(prompt)
        if not keywords:
            return None
            
        # Try each keyword with each knowledge source
        for keyword in keywords:
            # Try Semantic Scholar first
            knowledge = self.retrieve_from_semantic_scholar(keyword)
            if knowledge:
                self.update_last_time()
                return knowledge
                
            # Then try Wikipedia
            knowledge = self.retrieve_from_wikipedia(keyword)
            if knowledge:
                self.update_last_time()
                return knowledge
                
        return None

# --- Feedback Manager ---
class FeedbackManager:
    def __init__(self, config: NexusConfig, logger: LoggingManager):
        self.config = config
        self.logger = logger
        self.performance_history = deque(maxlen=config.performance_history_length)
        
    def save_feedback(self, prompt: str, response: str, rating: str):
        """Save feedback to disk"""
        with open(self.config.feedback_data_path, 'a') as f:
            feedback_entry = {
                'prompt': prompt,
                'response': response,
                'rating': rating.lower(),
                'timestamp': datetime.now().isoformat()
            }
            f.write(json.dumps(feedback_entry) + '\n')
            
        self.logger.info(f"Feedback saved: Rating='{rating}'")
        self.performance_history.append(1 if rating == 'good' else 0)
        
    def load_feedback(self) -> List[Dict[str, Any]]:
        """Load all feedback data from disk"""
        feedback_data = []
        if os.path.exists(self.config.feedback_data_path):
            with open(self.config.feedback_data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            feedback_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not parse feedback line: {line}")
                            
        return feedback_data
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Calculate performance statistics from history"""
        if not self.performance_history:
            return {"avg_rating": 0, "recent_trend": 0}
            
        avg_rating = sum(self.performance_history) / len(self.performance_history)
        
        # Calculate trend (positive or negative)
        if len(self.performance_history) >= 10:
            recent = list(self.performance_history)[-5:]
            older = list(self.performance_history)[-10:-5]
            trend = sum(recent) / len(recent) - sum(older) / len(older)
        else:
            trend = 0
            
        return {
            "avg_rating": avg_rating,
            "recent_trend": trend
        }
        
    def has_enough_feedback(self) -> bool:
        """Check if there's enough feedback for retraining"""
        return len(self.load_feedback()) >= self.config.min_feedback_for_retrain

# --- Model Manager ---
class ModelManager:
    def __init__(self, config: NexusConfig, logger: LoggingManager):
        self.config = config
        self.logger = logger
        self.tokenizer = None
        self.model = None
        
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize tokenizer and model from base model name"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
            
            # Ensure tokenizer has required special tokens
            special_tokens = {'pad_token': '[PAD]', 'sep_token': '[SEP]'}
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens(special_tokens)
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name
            ).to(self.config.device)
            
            # Resize token embeddings if we added tokens
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.logger.info(f"Model '{self.config.base_model_name}' loaded successfully on {self.config.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model '{self.config.base_model_name}': {e}")
            raise
            
    def load_from_pool(self, model_name: str) -> bool:
        """Load a model from the model pool"""
        model_path = os.path.join(self.config.model_pool_path, model_name)
        
        if not os.path.exists(model_path):
            self.logger.warning(f"Model '{model_name}' not found in pool")
            return False
            
        try:
            self.logger.info(f"Loading model from: '{model_path}'")
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.config.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model from '{model_path}': {e}")
            return False
            
    def save_to_pool(self, model_name: str):
        """Save current model to the model pool"""
        pool_path = os.path.join(self.config.model_pool_path, model_name)
        os.makedirs(pool_path, exist_ok=True)
        
        try:
            self.model.save_pretrained(pool_path)
            self.tokenizer.save_pretrained(pool_path)
            self.logger.info(f"Model '{model_name}' saved to pool: '{pool_path}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model to pool: {e}")
            return False
            
    def prepare_dataloaders(self, feedback_data: List[Dict[str, Any]]) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Prepare train and validation dataloaders from feedback data"""
        if not feedback_data:
            return None, None
            
        prompts = [entry['prompt'] for entry in feedback_data]
        responses = [entry['response'] for entry in feedback_data]
        
        # Split into train and validation sets
        train_prompts, val_prompts, train_responses, val_responses = train_test_split(
            prompts, responses, test_size=self.config.train_test_split, random_state=42
        )
        
        # Create datasets
        train_dataset = FeedbackDataset(
            train_prompts, train_responses, 
            self.tokenizer, self.config.max_input_length, 
            self.config.device
        )
        
        val_dataset = FeedbackDataset(
            val_prompts, val_responses, 
            self.tokenizer, self.config.max_input_length, 
            self.config.device
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        return train_dataloader, val_dataloader
        
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                total_loss += outputs.loss.item()
                
        avg_loss = total_loss / len(dataloader)
        self.model.train()
        return avg_loss
        
    def fine_tune(self, feedback_data: List[Dict[str, Any]], num_epochs: int = 3) -> Dict[str, float]:
        """Fine-tune model on feedback data"""
        train_dataloader, val_dataloader = self.prepare_dataloaders(feedback_data)
        
        if not train_dataloader or not val_dataloader:
            self.logger.warning("No data available for fine-tuning")
            return {"train_loss": float('inf'), "val_loss": float('inf')}
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        training_stats = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                epoch_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            # Calculate average loss for epoch
            avg_train_loss = epoch_loss / len(train_dataloader)
            
            # Evaluate on validation set
            val_loss = self.evaluate(val_dataloader)
            
            # Log performance
            log_message = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
            self.logger.info(log_message)
            self.logger.log_performance(log_message)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_name = f"{os.path.basename(self.config.base_model_name)}_finetuned_best"
                self.save_to_pool(model_name)
                
            # Store stats
            training_stats.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss
            })
        
        # Save final model
        model_name = f"{os.path.basename(self.config.base_model_name)}_finetuned_final"
        self.save_to_pool(model_name)
        
        # Return final stats
        return {
            "train_loss": training_stats[-1]["train_loss"],
            "val_loss": training_stats[-1]["val_loss"],
            "best_val_loss": best_val_loss
        }
    
    def generate_text(self, prompt: str, 
                    max_length: int = None, 
                    temperature: float = None,
                    top_p: float = None,
                    top_k: int = None) -> str:
        """Generate text based on a prompt"""
        # Use defaults if parameters not provided
        max_length = max_length or self.config.default_max_length
        temperature = temperature or self.config.default_temperature
        top_p = top_p or self.config.default_top_p
        top_k = top_k or self.config.default_top_k
        
        try:
            # Encode prompt
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=self.config.max_input_length, 
                truncation=True
            ).to(self.config.device)
            
            # Set model to eval mode for inference
            self.model.eval()
            
            # Generate
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Restore model to training mode
            self.model.train()
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return f"Error: An unexpected error occurred during text generation."

# --- Hyperparameter Optimization ---
class HyperparameterOptimizer:
    def __init__(self, config: NexusConfig, logger: LoggingManager, feedback_data: List[Dict[str, Any]]):
        self.config = config
        self.logger = logger
        self.feedback_data = feedback_data
        
    def _objective(self, trial):
        """Optuna objective function for optimization"""
        # Sample hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
        
        # Configure temporary model
        temp_config = NexusConfig(
            base_model_name=self.config.base_model_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            feedback_data_path=self.config.feedback_data_path,
            max_input_length=self.config.max_input_length,
            train_test_split=self.config.train_test_split,
            device=self.config.device
        )
        
        # Initialize model with temporary config
        model_manager = ModelManager(temp_config, self.logger)
        
        # Fine-tune and evaluate
        stats = model_manager.fine_tune(self.feedback_data, num_epochs=1)
        
        return stats["val_loss"]
        
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        if not self.feedback_data:
            self.logger.warning("No feedback data available for optimization")
            return {}
            
        self.logger.info("Starting hyperparameter optimization...")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(self._objective, n_trials=self.config.hyperopt_max_evals)
        
        best_params = study.best_params
        self.logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params

# --- Intent Predictor ---
class IntentPredictor:
    def __init__(self):
        # Keyword-based intent mapping (simple approach)
        self.intent_keywords = {
            "summarize": {"max_length": 150, "temperature": 0.7},
            "explain": {"temperature": 0.9},
            "compare": {"temperature": 0.8},
            "list": {"temperature": 0.6},
            "generate": {"temperature": 1.0},
            "concise": {"max_length": 100},
            "detailed": {"max_length": 500}
        }
        
    def predict_intent(self, prompt: str) -> Dict[str, Any]:
        """Predict user intent from prompt and return generation parameters"""
        prompt_lower = prompt.lower()
        
        # Default parameters
        params = {}
        
        # Check for keyword matches
        for keyword, settings in self.intent_keywords.items():
            if keyword in prompt_lower:
                params.update(settings)
                
        return params

# --- Main NexusLLM Class ---
class NexusLLM:
    def __init__(self):
        # Initialize configuration
        self.config = NexusConfig()
        
        # Initialize components
        self.logger = LoggingManager(self.config)
        self.model_manager = ModelManager(self.config, self.logger)
        self.feedback_manager = FeedbackManager(self.config, self.logger)
        self.knowledge_retriever = KnowledgeRetriever(self.config, self.logger)
        self.intent_predictor = IntentPredictor()
        
        # Counters and flags
        self.interaction_count = 0
        self._optimization_running = False
        
        self.logger.info("Omnitide Nexus LLM Initialized Successfully")
        
    def generate(self, prompt: str) -> str:
        """Generate text response for a prompt"""
        if not prompt or not isinstance(prompt, str):
            return "Error: Prompt cannot be empty and must be a string."
            
        try:
            # Knowledge retrieval and integration
            knowledge = self.knowledge_retriever.retrieve(prompt)
            enhanced_prompt = prompt
            if knowledge:
                enhanced_prompt = f"Context: '{knowledge}'. Respond to: '{prompt}'"
                self.logger.info("Enhanced prompt with external knowledge")
                
            # Predict user intent
            intent_params = self.intent_predictor.predict_intent(prompt)
            self.logger.info(f"Predicted intent parameters: {intent_params}")
            
            # Generate response
            response = self.model_manager.generate_text(
                enhanced_prompt,
                max_length=intent_params.get('max_length'),
                temperature=intent_params.get('temperature'),
                top_p=intent_params.get('top_p'),
                top_k=intent_params.get('top_k')
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            return f"Error: An unexpected error occurred. {str(e)}"
    
    def receive_feedback(self, prompt: str, response: str, rating: str):
        """Process user feedback"""
        if rating.lower() not in ['good', 'bad', 'neutral']:
            return False
            
        # Save feedback
        self.feedback_manager.save_feedback(prompt, response, rating.lower())
        self.interaction_count += 1
        
        # Check for training threshold
        if self.interaction_count % self.config.evaluation_interval == 0 and self.feedback_manager.has_enough_feedback():
            self._trigger_training()
            
        # Check for optimization threshold
        if (self.interaction_count % self.config.self_optimization_frequency == 0 and 
            self.feedback_manager.has_enough_feedback() and 
            not self._optimization_running):
            self._trigger_optimization()
            
        return True
    
    def _trigger_training(self):
        """Trigger model fine-tuning in a separate thread"""
        self.logger.info("Initiating training on collected feedback")
        
        def training_task():
            feedback_data = self.feedback_manager.load_feedback()
            stats = self.model_manager.fine_tune(feedback_data)
            self.logger.info(f"Training complete. Final training loss: {stats['train_loss']:.4f}, validation loss: {stats['val_loss']:.4f}")
            
        # Start training in a separate thread
        training_thread = threading.Thread(target=training_task)
        training_thread.daemon = True
        training_thread.start()
    
    def _trigger_optimization(self):
        """Trigger hyperparameter optimization in a separate thread"""
        self.logger.info("Initiating hyperparameter optimization")
        self._optimization_running = True
        
        def optimization_task():
            try:
                feedback_data = self.feedback_manager.load_feedback()
                optimizer = HyperparameterOptimizer(self.config, self.logger, feedback_data)
                best_params = optimizer.optimize()
                
                # Update configuration with best parameters
                if 'learning_rate' in best_params:
                    self.config.learning_rate = best_params['learning_rate']
                if 'weight_decay' in best_params:
                    self.config.weight_decay = best_params['weight_decay']
                if 'batch_size' in best_params:
                    self.config.batch_size = best_params['batch_size']
                    
                self.logger.info(f"Optimization complete. Updated config: learning_rate={self.config.learning_rate}, weight_decay={self.config.weight_decay}, batch_size={self.config.batch_size}")
            finally:
                self._optimization_running = False
                
        # Start optimization in a separate thread
        optimization_thread = threading.Thread(target=optimization_task)
        optimization_thread.daemon = True
        optimization_thread.start()
    
    def run(self):
        """Run interactive loop for the LLM"""
        self.logger.info("Starting interactive session")
        print("Omnitide Nexus LLM Interactive Session (Type 'exit' to quit)")
        
        while True:
            try:
                user_prompt = input("\nYour prompt: ")
                if user_prompt.lower() == 'exit':
                    print("Exiting Nexus LLM session. Goodbye!")
                    break
                    
                print("\nGenerating response...")
                start_time = time.time()
                generated_output = self.generate(user_prompt)
                generation_time = time.time() - start_time
                
                print(f"\nResponse (generated in {generation_time:.2f}s):\n{generated_output}")
                
                rating = input("\nRate the response (Good/Bad/Neutral): ").strip().lower()
                if rating in ['good', 'bad', 'neutral']:
                    self.receive_feedback(user_prompt, generated_output, rating)
                else:
                    print("Invalid feedback. Please use Good, Bad, or Neutral.")
                    
            except KeyboardInterrupt:
                print("\nReceived interrupt. Exiting...")
                break
            except Exception as e:
                self.logger.error(f"Error in interactive loop: {e}")
                print(f"An error occurred: {e}")

# --- Main Application Entry Point ---
if __name__ == "__main__":
    nexus_llm = NexusLLM()
    nexus_llm.run()
