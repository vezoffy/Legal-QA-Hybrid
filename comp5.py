import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize
import re
import time
from tqdm import tqdm
import torch.nn.functional as F
import datetime
 
# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
 
class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with Attention mechanism for extractive QA
    """
    def __init__(self, hidden_size, embedding_dim, num_layers=2, dropout=0.2):
        super(BiLSTMAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, 2)  # 2 outputs: start and end positions
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, embedding_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x, hidden)
        # lstm_out shape: (batch_size, seq_len, hidden_size*2)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out)
        # attention_scores shape: (batch_size, seq_len, 1)
        
        attention_weights = F.softmax(attention_scores, dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)
        
        # Apply attention weights
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        # context_vector shape: (batch_size, hidden_size*2)
        
        # Output layer - raw logits, not probabilities
        output = self.fc(context_vector)
        # output shape: (batch_size, 2)
        
        return output, attention_weights
 
 
class LegalQADataset(Dataset):
    """Dataset for training QA models with legal data"""
    def __init__(self, questions, contexts, answer_starts, answer_ends, tokenizer, max_length=512):
        self.questions = questions
        self.contexts = contexts
        self.answer_starts = answer_starts
        self.answer_ends = answer_ends
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer_start = self.answer_starts[idx]
        answer_end = self.answer_ends[idx]
        
        # Tokenize for BERT/T5
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add answer positions
        inputs['answer_start'] = answer_start
        inputs['answer_end'] = answer_end
        
        return inputs
 
 
class AnswerExtractorGenerator:
    def __init__(self, device=None, extractive_model_path=None, generative_model_path=None):
        """
        Initialize Answer Extraction and Generation Component
        
        Args:
            device (str): Device to use (cpu or cuda)
            extractive_model_path (str): Path to load the extractive model
            generative_model_path (str): Path to load the generative model
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizers and models
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Initialize extractive model
        self.extractive_model = None
        if extractive_model_path:
            self.load_extractive_model(extractive_model_path)
        
        # Initialize generative model (T5)
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.t5_model = None
        
        if generative_model_path:
            self.load_generative_model(generative_model_path)
        else:
            print("Loading default T5 model...")
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(self.device)
    
    def prepare_training_data(self, processed_data_path, test_size=0.2):
        """
        Prepare training data from processed dataset
        
        Args:
            processed_data_path (str): Path to processed data
            test_size (float): Proportion of data for testing
            
        Returns:
            tuple: Train and test data as DataLoader objects
        """
        # Load processed data
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if 'processed_df' in data:
            processed_df = pd.DataFrame(data['processed_df'])
        else:
            raise ValueError("Processed data not found in expected format")
        
        questions = processed_df['question'].tolist()
        answers = processed_df['answer'].tolist()
        
        # Simulate context by combining question with answer
        # In a real application, this would come from retrieved passages
        contexts = []
        answer_starts = []
        answer_ends = []
        
        for q, a in zip(questions, answers):
            # Create a simple context by joining question and answer
            # This is for demonstration; in production, use retrieved passages
            context = f"Question: {q} Answer: {a}"
            contexts.append(context)
            
            # Find the start position of answer in context
            start_pos = context.lower().find(a.lower())
            if start_pos >= 0:
                answer_starts.append(start_pos)
                answer_ends.append(start_pos + len(a))
            else:
                # Fallback if answer not found in context
                answer_starts.append(0)
                answer_ends.append(len(a))
        
        # Split data into train and test sets
        train_questions, test_questions, train_contexts, test_contexts, \
        train_starts, test_starts, train_ends, test_ends = train_test_split(
            questions, contexts, answer_starts, answer_ends, 
            test_size=test_size, random_state=42
        )
        
        # Create datasets
        train_dataset = LegalQADataset(
            train_questions, train_contexts, train_starts, train_ends, 
            self.bert_tokenizer
        )
        
        test_dataset = LegalQADataset(
            test_questions, test_contexts, test_starts, test_ends, 
            self.bert_tokenizer
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=8, shuffle=True
        )
        
        test_dataloader = DataLoader(
            test_dataset, batch_size=8
        )
        
        print(f"Prepared {len(train_dataset)} training samples and {len(test_dataset)} test samples")
        
        return train_dataloader, test_dataloader
    
    def train_extractive_model(self, train_dataloader, test_dataloader, 
                               hidden_size=256, embedding_dim=768, num_layers=2,
                               num_epochs=5, learning_rate=3e-5):
        """
        Train the BiLSTM with Attention model for extractive QA with progress tracking
        
        Args:
            train_dataloader (DataLoader): Training data
            test_dataloader (DataLoader): Test data
            hidden_size (int): Size of LSTM hidden layer
            embedding_dim (int): Size of word embeddings
            num_layers (int): Number of LSTM layers
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            
        Returns:
            BiLSTMAttention: Trained model
        """
        # Initialize model
        model = BiLSTMAttention(
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss function - use MSELoss instead of CrossEntropyLoss for regression task
        criterion = nn.MSELoss()
        
        # Calculate total steps for progress tracking
        total_train_steps = len(train_dataloader)
        total_test_steps = len(test_dataloader)
        
        print(f"Starting training: {num_epochs} epochs, {total_train_steps} steps per epoch")
        print(f"Training on device: {self.device}")
        
        # Track training metrics
        all_train_losses = []
        all_test_losses = []
        start_time = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # TRAINING PHASE
            model.train()
            train_loss = 0
            batch_losses = []
            
            # Create progress bar for training
            train_progress_bar = tqdm(enumerate(train_dataloader), 
                                     total=total_train_steps, 
                                     desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                                     leave=True)
            
            for batch_idx, batch in train_progress_bar:
                # Get input embeddings from BERT
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    bert_outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    bert_embeddings = bert_outputs.last_hidden_state
                
                # Get answer positions and convert to float for MSELoss
                start_positions = batch['answer_start'].float().to(self.device)
                end_positions = batch['answer_end'].float().to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output, _ = model(bert_embeddings)
                
                # Calculate loss with MSELoss (requires float targets)
                start_loss = criterion(output[:, 0], start_positions)
                end_loss = criterion(output[:, 1], end_positions)
                loss = start_loss + end_loss
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
                train_loss += batch_loss
                batch_losses.append(batch_loss)
                
                # Update progress bar with current loss
                train_progress_bar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'avg_loss': f"{train_loss/(batch_idx+1):.4f}"
                })
            
            # Calculate average training loss
            avg_train_loss = train_loss / total_train_steps
            all_train_losses.append(avg_train_loss)
            
            # EVALUATION PHASE
            model.eval()
            test_loss = 0
            
            # Create progress bar for evaluation
            test_progress_bar = tqdm(enumerate(test_dataloader), 
                                    total=total_test_steps, 
                                    desc=f"Epoch {epoch+1}/{num_epochs} [Test]",
                                    leave=True)
            
            with torch.no_grad():
                for batch_idx, batch in test_progress_bar:
                    # Get input embeddings from BERT
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    bert_outputs = self.bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    bert_embeddings = bert_outputs.last_hidden_state
                    
                    # Get answer positions as float for MSELoss
                    start_positions = batch['answer_start'].float().to(self.device)
                    end_positions = batch['answer_end'].float().to(self.device)
                    
                    # Forward pass
                    output, _ = model(bert_embeddings)
                    
                    # Calculate loss with MSELoss
                    start_loss = criterion(output[:, 0], start_positions)
                    end_loss = criterion(output[:, 1], end_positions)
                    loss = start_loss + end_loss
                    
                    batch_loss = loss.item()
                    test_loss += batch_loss
                    
                    # Update progress bar
                    test_progress_bar.set_postfix({
                        'loss': f"{batch_loss:.4f}",
                        'avg_loss': f"{test_loss/(batch_idx+1):.4f}"
                    })
            
            # Calculate average test loss
            avg_test_loss = test_loss / total_test_steps
            all_test_losses.append(avg_test_loss)
            
            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time
            total_duration = time.time() - start_time
            
            # Format as hours:minutes:seconds
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_duration)))
            total_time_str = str(datetime.timedelta(seconds=int(total_duration)))
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Test Loss: {avg_test_loss:.4f}")
            print(f"  Epoch Time: {epoch_time_str}")
            print(f"  Total Time: {total_time_str}")
            
            # Estimate remaining time
            if epoch < num_epochs - 1:
                avg_epoch_time = total_duration / (epoch + 1)
                remaining_epochs = num_epochs - (epoch + 1)
                est_remaining_time = avg_epoch_time * remaining_epochs
                est_remaining_str = str(datetime.timedelta(seconds=int(est_remaining_time)))
                print(f"  Estimated Time Remaining: {est_remaining_str}")
            
            print("-" * 50)
        
        # Print training summary
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"\nTraining completed in {total_time_str}")
        print(f"Final Train Loss: {all_train_losses[-1]:.4f}")
        print(f"Final Test Loss: {all_test_losses[-1]:.4f}")
        
        # Save the trained model
        self.extractive_model = model
        
        return model
    
    def save_extractive_model(self, path):
        """
        Save the trained extractive model
        
        Args:
            path (str): Path to save the model
        """
        if self.extractive_model is None:
            raise ValueError("No extractive model available to save")
            
        torch.save({
            'model_state_dict': self.extractive_model.state_dict(),
            'hidden_size': self.extractive_model.hidden_size,
            'num_layers': self.extractive_model.num_layers,
            'dropout': self.extractive_model.dropout
        }, path)
        
        print(f"Extractive model saved to {path}")
    
    def load_extractive_model(self, path):
        """
        Load a trained extractive model
        
        Args:
            path (str): Path to load the model from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            model = BiLSTMAttention(
                hidden_size=checkpoint['hidden_size'],
                embedding_dim=768,  # BERT embedding size
                num_layers=checkpoint['num_layers'],
                dropout=checkpoint['dropout']
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.extractive_model = model
            print(f"Extractive model loaded from {path}")
            
        except Exception as e:
            print(f"Error loading extractive model: {str(e)}")
    
    def fine_tune_t5(self, train_dataloader, test_dataloader, num_epochs=3, learning_rate=5e-5):
        """
        Fine-tune T5 model for answer generation with progress tracking
        
        Args:
            train_dataloader (DataLoader): Training data
            test_dataloader (DataLoader): Test data
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            
        Returns:
            T5ForConditionalGeneration: Fine-tuned model
        """
        # Initialize model if not already loaded
        if self.t5_model is None:
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(self.device)
        
        # Initialize optimizer
        optimizer = optim.AdamW(self.t5_model.parameters(), lr=learning_rate)
        
        # Calculate total steps for progress tracking
        total_train_steps = len(train_dataloader)
        total_test_steps = len(test_dataloader)
        
        print(f"Starting T5 fine-tuning: {num_epochs} epochs, {total_train_steps} steps per epoch")
        print(f"Training on device: {self.device}")
        
        # Track training metrics
        all_train_losses = []
        all_test_losses = []
        start_time = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # TRAINING PHASE
            self.t5_model.train()
            train_loss = 0
            
            # Create progress bar for training
            train_progress_bar = tqdm(enumerate(train_dataloader), 
                                     total=total_train_steps, 
                                     desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                                     leave=True)
            
            for batch_idx, batch in train_progress_bar:
                # Prepare inputs for T5
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Create target sequence (use the answer as target)
                # In real application, extract answer text using answer_start and answer_end
                # Here we'll use a simplified approach
                target_text = batch['answer_start'].to(self.device)  # Placeholder
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.t5_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target_text
                )
                
                loss = outputs.loss
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
                train_loss += batch_loss
                
                # Update progress bar with current loss
                train_progress_bar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'avg_loss': f"{train_loss/(batch_idx+1):.4f}"
                })
            
            # Calculate average training loss
            avg_train_loss = train_loss / total_train_steps
            all_train_losses.append(avg_train_loss)
            
            # EVALUATION PHASE
            self.t5_model.eval()
            test_loss = 0
            
            # Create progress bar for evaluation
            test_progress_bar = tqdm(enumerate(test_dataloader), 
                                    total=total_test_steps, 
                                    desc=f"Epoch {epoch+1}/{num_epochs} [Test]",
                                    leave=True)
            
            with torch.no_grad():
                for batch_idx, batch in test_progress_bar:
                    # Prepare inputs for T5
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Create target sequence
                    target_text = batch['answer_start'].to(self.device)  # Placeholder
                    
                    # Forward pass
                    outputs = self.t5_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=target_text
                    )
                    
                    loss = outputs.loss
                    batch_loss = loss.item()
                    test_loss += batch_loss
                    
                    # Update progress bar
                    test_progress_bar.set_postfix({
                        'loss': f"{batch_loss:.4f}",
                        'avg_loss': f"{test_loss/(batch_idx+1):.4f}"
                    })
            
            # Calculate average test loss
            avg_test_loss = test_loss / total_test_steps
            all_test_losses.append(avg_test_loss)
            
            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time
            total_duration = time.time() - start_time
            
            # Format as hours:minutes:seconds
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_duration)))
            total_time_str = str(datetime.timedelta(seconds=int(total_duration)))
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Test Loss: {avg_test_loss:.4f}")
            print(f"  Epoch Time: {epoch_time_str}")
            print(f"  Total Time: {total_time_str}")
            
            # Estimate remaining time
            if epoch < num_epochs - 1:
                avg_epoch_time = total_duration / (epoch + 1)
                remaining_epochs = num_epochs - (epoch + 1)
                est_remaining_time = avg_epoch_time * remaining_epochs
                est_remaining_str = str(datetime.timedelta(seconds=int(est_remaining_time)))
                print(f"  Estimated Time Remaining: {est_remaining_str}")
            
            print("-" * 50)
        
        # Print training summary
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"\nT5 fine-tuning completed in {total_time_str}")
        print(f"Final Train Loss: {all_train_losses[-1]:.4f}")
        print(f"Final Test Loss: {all_test_losses[-1]:.4f}")
        
        return self.t5_model
    
    def save_generative_model(self, path):
        """
        Save the fine-tuned T5 model
        
        Args:
            path (str): Path to save the model
        """
        if self.t5_model is None:
            raise ValueError("No generative model available to save")
            
        self.t5_model.save_pretrained(path)
        self.t5_tokenizer.save_pretrained(path)
        
        print(f"Generative model saved to {path}")
    
    def load_generative_model(self, path):
        """
        Load a fine-tuned T5 model
        
        Args:
            path (str): Path to load the model from
        """
        try:
            self.t5_model = T5ForConditionalGeneration.from_pretrained(path).to(self.device)
            self.t5_tokenizer = T5Tokenizer.from_pretrained(path)
            
            print(f"Generative model loaded from {path}")
            
        except Exception as e:
            print(f"Error loading generative model: {str(e)}")
            print("Loading default T5 model instead...")
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small').to(self.device)
    
    def extract_answer(self, question, context, confidence_threshold=0.5):
        """
        Extract answer from context using BiLSTM with Attention
        
        Args:
            question (str): User question
            context (str): Retrieved passage
            confidence_threshold (float): Threshold for extraction confidence
            
        Returns:
            dict: Extracted answer with confidence score
        """
        if self.extractive_model is None:
            return {'answer': None, 'confidence': 0.0, 'message': 'Extractive model not available'}
        
        # Tokenize input
        encoding = self.bert_tokenizer(
            question,
            context,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get BERT embeddings
        with torch.no_grad():
            bert_outputs = self.bert_model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            
            bert_embeddings = bert_outputs.last_hidden_state
        
        # Get predictions from extractive model
        self.extractive_model.eval()
        with torch.no_grad():
            output, attention_weights = self.extractive_model(bert_embeddings)
            
            # Get start and end positions
            start_logits = output[0, 0].item()
            end_logits = output[0, 1].item()
            
            # Convert logits to probabilities
            start_prob = torch.sigmoid(torch.tensor(start_logits)).item()
            end_prob = torch.sigmoid(torch.tensor(end_logits)).item()
            
            confidence = (start_prob + end_prob) / 2
            
            # If confidence is below threshold, return no answer
            if confidence < confidence_threshold:
                return {
                    'answer': None,
                    'confidence': confidence,
                    'message': 'Low confidence in extraction'
                }
            
            # Get token with highest attention weight
            attention = attention_weights.squeeze().cpu().numpy()
            max_attention_idx = np.argmax(attention)
            
            # Extract tokens around the highest attention point
            # Convert to original tokens
            tokens = self.bert_tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
            
            # Take a window around the highest attention token
            window_size = 5
            start_idx = max(0, max_attention_idx - window_size)
            end_idx = min(len(tokens), max_attention_idx + window_size)
            
            # Get the extracted tokens
            extracted_tokens = tokens[start_idx:end_idx]
            
            # Convert back to text
            extracted_text = self.bert_tokenizer.convert_tokens_to_string(extracted_tokens)
            
            # Clean up the extracted text
            extracted_text = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', extracted_text).strip()
            
            return {
                'answer': extracted_text,
                'confidence': confidence,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
    
    def generate_answer(self, question, context, max_length=100):
        """
        Generate answer using T5 model
        
        Args:
            question (str): User question
            context (str): Retrieved passage
            max_length (int): Maximum answer length
            
        Returns:
            dict: Generated answer
        """
        if self.t5_model is None:
            return {'answer': None, 'message': 'Generative model not available'}
        
        # Prepare input for T5
        input_text = f"question: {question} context: {context}"
        
        # Tokenize input
        input_ids = self.t5_tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        # Generate answer
        self.t5_model.eval()
        with torch.no_grad():
            output_ids = self.t5_model.generate(
                input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the output
            answer = self.t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            return {
                'answer': answer,
                'model': 't5'
            }
    
    def get_best_answer(self, question, contexts, min_confidence=0.6):
        """
        Get the best answer using both extractive and generative approaches
        
        Args:
            question (str): User question
            contexts (list): List of retrieved passages
            min_confidence (float): Minimum confidence for extractive answers
            
        Returns:
            dict: Best answer with metadata
        """
        if not contexts:
            return {
                'answer': None,
                'message': 'No contexts provided',
                'method': None
            }
        
        extractive_results = []
        
        # Try extractive approach on each context
        for context in contexts:
            result = self.extract_answer(question, context, min_confidence)
            if result['answer']:
                extractive_results.append({
                    'answer': result['answer'],
                    'confidence': result['confidence'],
                    'context': context
                })
        
        # Sort extractive results by confidence
        extractive_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # If we have good extractive answers, return the best one
        if extractive_results and extractive_results[0]['confidence'] >= min_confidence:
            return {
                'answer': extractive_results[0]['answer'],
                'confidence': extractive_results[0]['confidence'],
                'context': extractive_results[0]['context'],
                'method': 'extractive'
            }
        
        # No good extractive answer, try generative approach
        # Combine contexts for better context
        combined_context = " ".join(contexts[:3])  # Use top 3 contexts
        
        generative_result = self.generate_answer(question, combined_context)
        
        if generative_result['answer']:
            return {
                'answer': generative_result['answer'],
                'method': 'generative',
                'context': combined_context
            }
        
        # No good answer from either method
        return {
            'answer': None,
            'message': 'Failed to find or generate answer',
            'method': None
        }
 
# Example usage
"""
# Initialize the Answer Extractor/Generator
answer_component = AnswerExtractorGenerator()
 
# Load processed data and prepare for training
train_loader, test_loader = answer_component.prepare_training_data('legal_qa_processed.json')
 
# Train extractive model
extractive_model = answer_component.train_extractive_model(
    train_loader, test_loader, num_epochs=3
)
 
# Save extractive model
answer_component.save_extractive_model('extractive_model.pt')
 
# Fine-tune T5 for answer generation
t5_model = answer_component.fine_tune_t5(
    train_loader, test_loader, num_epochs=2
)
 
# Save generative model
answer_component.save_generative_model('generative_model')
 
# Test with sample question and contexts
question = "What are the fundamental rights in the Indian Constitution?"
contexts = [
    "The Constitution of India provides six fundamental rights to Indian citizens. These are: Right to Equality, Right to Freedom, Right against Exploitation, Right to Freedom of Religion, Cultural and Educational Rights, and Right to Constitutional Remedies.",
    "Fundamental Rights is a charter of rights contained in the Constitution of India. It guarantees civil liberties such that all Indians can lead their lives in peace and harmony as citizens of India.",
    "The Fundamental Rights in India are contained in Part III of the Constitution from Articles 12 to 35."
]
 
# Get best answer
result = answer_component.get_best_answer(question, contexts)
print(f"Question: {question}")
print(f"Answer: {result['answer']}")
print(f"Method: {result['method']}")
if 'confidence' in result:
    print(f"Confidence: {result['confidence']:.2f}")
"""