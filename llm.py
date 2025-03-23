#!/usr/bin/env python
"""
NexusLLM - Automated Language Model with Feedback-Based Optimization
"""
import os
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import optuna
import requests
from bs4 import BeautifulSoup

# --- Configuration ---
@dataclass
class NexusConfig:
    # Model configuration
    base_model_name: str = "google/flan-t5-small"  # Using smaller model for testing
    
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
    max_input_length: int = 512  # Reduced for efficiency
    train_test_split: float = 0.2
    
    # Optimization parameters
    self_optimization_frequency: int = 100
    hyperopt_max_evals: int = 5  # Reduced for testing
    optimization_metric: str = "val_loss"
    min_feedback_for_retrain: int = 10  # Reduced for testing
    performance_history_length: int = 30
    
    # Knowledge retrieval
    knowledge_apis: Dict[str, str] = field(default_factory=lambda: {
        "wikipedia": "https://en.wikipedia.org/w/api.php"
    })
    knowledge_update_frequency: int = 3600
    
    # Hardware settings
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Generation parameters
    default_max_length: int = 300
    default_temperature: float = 0.7
    default_top_p: float = 0.85
    default_top_k: int = 50
    
    def __post_init__(self):
        os.makedirs(self.model_pool_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.performance_log_path) or '.', exist_ok=True)

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
        sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token else " "
        full_text = self.prompts[idx] + sep_token + self.responses[idx]
        
        # Handle different tokenizer return formats
        inputs = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        item = {k: v.squeeze().to(self.device) for k, v in inputs.items()}
        item['labels'] = item['input_ids'].clone()
        
        return item

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
        words = text.lower().split()
        # Filter out common stopwords
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "about", "like"}
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        return keywords[:num_keywords]
        
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
            
        # Try each keyword with Wikipedia
        for keyword in keywords:
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
            special_tokens = {}
            if self.tokenizer.pad_token is None:
                special_tokens['pad_token'] = '[PAD]'
            if self.tokenizer.sep_token is None:
                special_tokens['sep_token'] = '[SEP]'
                
            if special_tokens:
                self.tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
                
            # Load model configuration
            model_config = AutoConfig.from_pretrained(self.config.base_model_name)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                config=model_config
            ).to(self.config.device)
            
            # Resize token embeddings if we added tokens
            if special_tokens:
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
            with torch.no_grad():
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
            
            # Clean up response
            response = generated_text.replace(prompt, "").strip()
            if not response:
                response = generated_text  # Fallback to full text if prompt stripping removes everything
            
            # Restore model to training mode
            self.model.train()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return f"Error: An unexpected error occurred during text generation: {str(e)}"

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
        
        self.logger.info("NexusLLM Initialized Successfully")
        
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
        print("NexusLLM Interactive Session (Type 'exit' to quit)")
        
        while True:
            try:
                user_prompt = input("\nYour prompt: ")
                if user_prompt.lower() == 'exit':
                    print("Exiting NexusLLM session. Goodbye!")
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
                
#!/usr/bin/env python3
"""
Nexus LLM Dependencies Installer
-------------------------------
This utility installs all required dependencies for the Nexus LLM system.
"""

import subprocess
import sys
import os
import platform
import shutil
from typing import List, Dict, Tuple, Optional
import argparse

# Configuration
REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "scikit-learn",
    "optuna",
    "requests",
    "beautifulsoup4",
    "numpy",
    "dataclasses"
]

CUDA_PACKAGES = {
    "torch": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
}

class InstallerConfig:
    def __init__(self):
        self.verbose = False
        self.cuda = False
        self.venv_path = "nexus_env"
        self.upgrade = False
        self.requirements_file = "requirements.txt"
        self.skip_cuda_check = False
        self.force_cpu = False

class ColorOutput:
    """Class to handle colored terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def disable_colors():
        """Disable colors for non-compatible terminals"""
        ColorOutput.HEADER = ''
        ColorOutput.BLUE = ''
        ColorOutput.GREEN = ''
        ColorOutput.YELLOW = ''
        ColorOutput.RED = ''
        ColorOutput.ENDC = ''
        ColorOutput.BOLD = ''
        ColorOutput.UNDERLINE = ''
    
    @staticmethod
    def success(msg: str) -> str:
        return f"{ColorOutput.GREEN}✓ {msg}{ColorOutput.ENDC}"
    
    @staticmethod
    def error(msg: str) -> str:
        return f"{ColorOutput.RED}✗ {msg}{ColorOutput.ENDC}"
    
    @staticmethod
    def warning(msg: str) -> str:
        return f"{ColorOutput.YELLOW}⚠ {msg}{ColorOutput.ENDC}"
    
    @staticmethod
    def info(msg: str) -> str:
        return f"{ColorOutput.BLUE}→ {msg}{ColorOutput.ENDC}"
    
    @staticmethod
    def header(msg: str) -> str:
        return f"{ColorOutput.HEADER}{ColorOutput.BOLD}{msg}{ColorOutput.ENDC}"

class DependencyInstaller:
    """Handles the installation of Python dependencies for Nexus LLM"""
    
    def __init__(self, config: InstallerConfig):
        self.config = config
        self.pip_path = None
        self.python_path = None
        
        # Initialize colors
        if platform.system() == "Windows" and not self._is_ansi_terminal():
            ColorOutput.disable_colors()
    
    def _is_ansi_terminal(self) -> bool:
        """Check if terminal supports ANSI colors"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def print_header(self):
        """Print installer header"""
        print("\n" + "=" * 60)
        print(ColorOutput.header("NEXUS LLM DEPENDENCY INSTALLER"))
        print("=" * 60 + "\n")
    
    def print_step(self, step: str, total_steps: int, current_step: int):
        """Print current installation step"""
        print(f"\n{ColorOutput.BOLD}[{current_step}/{total_steps}] {step}{ColorOutput.ENDC}")
    
    def check_prerequisites(self) -> bool:
        """Check if prerequisites are met"""
        # Check Python version
        python_version = sys.version.split()[0]
        min_version = "3.8.0"
        
        if self._version_is_less(python_version, min_version):
            print(ColorOutput.error(f"Python version {python_version} detected. Nexus LLM requires Python {min_version} or higher."))
            return False
        
        print(ColorOutput.success(f"Python {python_version} detected."))
        
        # Check pip installation
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
            print(ColorOutput.success("Pip is installed."))
        except (subprocess.SubprocessError, FileNotFoundError):
            print(ColorOutput.error("Pip is not installed. Please install pip first."))
            return False
        
        # Check GPU availability if not skipped
        if self.config.cuda and not self.config.skip_cuda_check and not self.config.force_cpu:
            has_cuda = self._check_cuda()
            if has_cuda:
                print(ColorOutput.success("CUDA is available. Will install GPU-accelerated packages."))
            else:
                print(ColorOutput.warning("CUDA not detected. Falling back to CPU-only packages."))
                self.config.cuda = False
        
        return True
    
    def _version_is_less(self, current: str, minimum: str) -> bool:
        """Compare version strings"""
        current_parts = list(map(int, current.split('.')))
        minimum_parts = list(map(int, minimum.split('.')))
        
        for i in range(max(len(current_parts), len(minimum_parts))):
            current_part = current_parts[i] if i < len(current_parts) else 0
            minimum_part = minimum_parts[i] if i < len(minimum_parts) else 0
            
            if current_part < minimum_part:
                return True
            elif current_part > minimum_part:
                return False
        
        return False
    
    def _check_cuda(self) -> bool:
        """Check for CUDA availability"""
        try:
            # Try to import torch and check CUDA
            result = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if "True" in result.stdout:
                return True
            
            # Alternative: Check for NVIDIA GPU
            if platform.system() == "Windows":
                try:
                    nvidia_smi = subprocess.run(["where", "nvidia-smi"], capture_output=True, text=True, check=False)
                    if nvidia_smi.returncode == 0:
                        return True
                except FileNotFoundError:
                    pass
            else:
                try:
                    nvidia_smi = subprocess.run(["which", "nvidia-smi"], capture_output=True, text=True, check=False)
                    if nvidia_smi.returncode == 0:
                        return True
                except FileNotFoundError:
                    pass
                    
            return False
            
        except Exception:
            return False
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment for Nexus LLM"""
        if os.path.exists(self.config.venv_path):
            print(ColorOutput.warning(f"Virtual environment already exists at '{self.config.venv_path}'."))
            if not self.config.upgrade:
                choice = input("Do you want to recreate it? (y/n): ").strip().lower()
                if choice != 'y':
                    # Use existing environment
                    print(ColorOutput.info("Using existing virtual environment."))
                    self._setup_venv_paths()
                    return True
            
            # Delete existing environment
            print(ColorOutput.info(f"Removing existing virtual environment at '{self.config.venv_path}'..."))
            try:
                shutil.rmtree(self.config.venv_path)
            except Exception as e:
                print(ColorOutput.error(f"Failed to remove existing environment: {str(e)}"))
                return False
        
        # Create new environment
        try:
            print(ColorOutput.info(f"Creating virtual environment at '{self.config.venv_path}'..."))
            subprocess.run([sys.executable, "-m", "venv", self.config.venv_path], check=True)
            print(ColorOutput.success("Virtual environment created successfully."))
            
            self._setup_venv_paths()
            return True
            
        except subprocess.SubprocessError as e:
            print(ColorOutput.error(f"Failed to create virtual environment: {str(e)}"))
            return False
    
    def _setup_venv_paths(self):
        """Set up paths to Python and pip in the virtual environment"""
        if platform.system() == "Windows":
            self.python_path = os.path.join(self.config.venv_path, "Scripts", "python.exe")
            self.pip_path = os.path.join(self.config.venv_path, "Scripts", "pip.exe")
        else:
            self.python_path = os.path.join(self.config.venv_path, "bin", "python")
            self.pip_path = os.path.join(self.config.venv_path, "bin", "pip")
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip in the virtual environment"""
        try:
            print(ColorOutput.info("Upgrading pip to the latest version..."))
            subprocess.run([self.pip_path, "install", "--upgrade", "pip"], check=True)
            print(ColorOutput.success("Pip upgraded successfully."))
            return True
        except subprocess.SubprocessError as e:
            print(ColorOutput.error(f"Failed to upgrade pip: {str(e)}"))
            return False
    
    def create_requirements_file(self) -> bool:
        """Create requirements.txt file"""
        try:
            print(ColorOutput.info(f"Creating {self.config.requirements_file}..."))
            
            requirements = []
            for package in REQUIRED_PACKAGES:
                if self.config.cuda and package in CUDA_PACKAGES:
                    requirements.append(CUDA_PACKAGES[package])
                else:
                    requirements.append(package)
            
            with open(self.config.requirements_file, "w") as f:
                for req in requirements:
                    f.write(f"{req}\n")
            
            print(ColorOutput.success(f"{self.config.requirements_file} created successfully."))
            return True
        except Exception as e:
            print(ColorOutput.error(f"Failed to create requirements file: {str(e)}"))
            return False
    
    def install_requirements(self) -> bool:
        """Install requirements from requirements.txt"""
        try:
            print(ColorOutput.info(f"Installing packages from {self.config.requirements_file}..."))
            
            install_cmd = [self.pip_path, "install", "-r", self.config.requirements_file]
            if self.config.upgrade:
                install_cmd.append("--upgrade")
            
            process = subprocess.Popen(
                install_cmd,
                stdout=subprocess.PIPE if not self.config.verbose else None,
                stderr=subprocess.PIPE if not self.config.verbose else None,
                universal_newlines=True
            )
            
            if self.config.verbose:
                print("\n")
            else:
                print(ColorOutput.info("Installing packages (this may take a while)..."))
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode != 0:
                if not self.config.verbose and process.stderr:
                    print(ColorOutput.error(f"Installation output: {process.stderr.read()}"))
                print(ColorOutput.error(f"Failed to install packages. Return code: {process.returncode}"))
                return False
            
            print(ColorOutput.success("All packages installed successfully."))
            return True
            
        except Exception as e:
            print(ColorOutput.error(f"Failed to install packages: {str(e)}"))
            return False
    
    def verify_installation(self) -> bool:
        """Verify package installation by importing them"""
        print(ColorOutput.info("Verifying package installation..."))
        
        imports_to_check = {
            "torch": "import torch; print(f'PyTorch {torch.__version__} installed (\\'GPU available: {torch.cuda.is_available()}\\')')",
            "transformers": "import transformers; print(f'Transformers {transformers.__version__} installed')",
            "sklearn": "import sklearn; print(f'Scikit-learn {sklearn.__version__} installed')",
            "optuna": "import optuna; print(f'Optuna {optuna.__version__} installed')",
            "requests": "import requests; print(f'Requests {requests.__version__} installed')",
            "bs4": "import bs4; print(f'BeautifulSoup4 {bs4.__version__} installed')",
            "numpy": "import numpy as np; print(f'NumPy {np.__version__} installed')",
            "dataclasses": "import dataclasses; print('Dataclasses module installed')"
        }
        
        all_successful = True
        
        for package, import_code in imports_to_check.items():
            try:
                result = subprocess.run(
                    [self.python_path, "-c", import_code],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    print(ColorOutput.success(result.stdout.strip()))
                else:
                    print(ColorOutput.error(f"Failed to import {package}: {result.stderr.strip()}"))
                    all_successful = False
                    
            except Exception as e:
                print(ColorOutput.error(f"Error checking {package}: {str(e)}"))
                all_successful = False
        
        if all_successful:
            print(ColorOutput.success("\nAll packages verified successfully!"))
        else:
            print(ColorOutput.warning("\nSome packages could not be verified. Please check the output above."))
        
        return all_successful
    
    def show_activation_instructions(self):
        """Show instructions for activating the virtual environment"""
        print("\n" + "=" * 60)
        print(ColorOutput.header("INSTALLATION COMPLETE"))
        print("=" * 60)
        
        print("\nTo activate the Nexus LLM environment:\n")
        
        if platform.system() == "Windows":
            print(ColorOutput.info(f"  {self.config.venv_path}\\Scripts\\activate"))
        else:
            print(ColorOutput.info(f"  source {self.config.venv_path}/bin/activate"))
        
        print("\nAfter activation, you can run Nexus LLM from your project directory.")
        print("=" * 60 + "\n")
    
    def run(self) -> bool:
        """Run the complete installation process"""
        self.print_header()
        
        steps = [
            ("Checking prerequisites", self.check_prerequisites),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Upgrading pip", self.upgrade_pip),
            ("Creating requirements file", self.create_requirements_file),
            ("Installing packages", self.install_requirements),
            ("Verifying installation", self.verify_installation)
        ]
        
        for i, (step_name, step_func) in enumerate(steps, 1):
            self.print_step(step_name, len(steps), i)
            if not step_func():
                print(ColorOutput.error(f"\nInstallation failed at step {i}: {step_name}"))
                return False
        
        self.show_activation_instructions()
        return True

def parse_arguments() -> InstallerConfig:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Nexus LLM Dependencies Installer")
    
    parser.add_argument("--venv", dest="venv_path", default="nexus_env",
                        help="Virtual environment path (default: nexus_env)")
    
    parser.add_argument("--cuda", action="store_true", 
                        help="Install CUDA-enabled packages")
    
    parser.add_argument("--cpu", action="store_true", dest="force_cpu",
                        help="Force CPU-only packages even if CUDA is available")
    
    parser.add_argument("--upgrade", action="store_true",
                        help="Upgrade existing packages")
    
    parser.add_argument("--skip-cuda-check", action="store_true",
                        help="Skip CUDA availability check")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output during installation")
    
    args = parser.parse_args()
    
    config = InstallerConfig()
    config.venv_path = args.venv_path
    config.cuda = args.cuda
    config.upgrade = args.upgrade
    config.verbose = args.verbose
    config.skip_cuda_check = args.skip_cuda_check
    config.force_cpu = args.force_cpu
    
    return config

def main():
    """Main function"""
    config = parse_arguments()
    installer = DependencyInstaller(config)
    success = installer.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
