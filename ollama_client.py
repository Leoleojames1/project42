#!/usr/bin/env python3
"""
Ollama LLM integration module for Project 42
Handles communication with Ollama API for LLM inference
"""

import ollama
import asyncio
import logging


class OllamaManager:
    """
    Manages communication with Ollama API for LLM inference
    
    Features:
    - List available models
    - Chat completion with context handling
    - Streaming responses for real-time feedback
    """
    
    def __init__(self):
        """Initialize Ollama manager"""
        # Set up logging
        self.logger = logging.getLogger("OllamaManager")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def get_available_models(self):
        """
        Get list of available models from Ollama
        
        Returns:
            list: List of model names
        """
        try:
            self.logger.info("Fetching available models from Ollama")
            result = ollama.list()
            
            if 'models' in result:
                models = [model['name'] for model in result['models']]
                self.logger.info(f"Found {len(models)} models: {', '.join(models)}")
                return models
            
            self.logger.warning("No models found in Ollama response")
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting models: {str(e)}")
            return []
    
    def get_model_info(self, model_name):
        """
        Get detailed information about a specific model
        
        Args:
            model_name (str): Name of the model to get info for
            
        Returns:
            dict: Model information
        """
        try:
            return ollama.show(model_name)
        except Exception as e:
            self.logger.error(f"Error getting model info for {model_name}: {str(e)}")
            return {}
    
    def chat_completion(self, model, messages, stream=False):
        """
        Get chat completion from Ollama
        
        Args:
            model (str): Model name
            messages (list): List of message dictionaries
            stream (bool): Whether to stream the response
            
        Returns:
            dict or generator: Chat completion response
        """
        try:
            self.logger.info(f"Sending chat request to model: {model}")
            
            # Format messages according to Ollama API requirements
            formatted_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                # Ensure roles are supported by Ollama
                if role not in ['user', 'assistant', 'system']:
                    role = 'user'
                    
                formatted_messages.append({
                    'role': role,
                    'content': content
                })
            
            # Send request to Ollama
            return ollama.chat(
                model=model,
                messages=formatted_messages,
                stream=stream
            )
            
        except Exception as e:
            self.logger.error(f"Error in chat completion: {str(e)}")
            
            # Return error message
            if stream:
                # For streaming, return a generator that yields the error
                def error_generator():
                    yield {"message": {"content": f"Error: {str(e)}"}}
                return error_generator()
            else:
                # For non-streaming, return error dict
                return {"message": {"content": f"Error: {str(e)}"}}
    
    async def async_chat_completion(self, model, messages, stream=False):
        """
        Async version of chat completion
        
        Args:
            model (str): Model name
            messages (list): List of message dictionaries
            stream (bool): Whether to stream the response
            
        Returns:
            dict or generator: Chat completion response
        """
        # Wrap the synchronous function in an async executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.chat_completion(model, messages, stream)
        )
    
    def generate_chat_prompt(self, model, prompt, system_prompt=None, chat_history=None):
        """
        Generate a chat prompt with optional system prompt and history
        
        Args:
            model (str): Model name
            prompt (str): User prompt
            system_prompt (str): Optional system prompt
            chat_history (list): Optional chat history
            
        Returns:
            dict: Response from Ollama
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
        
        # Add user prompt
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        return self.chat_completion(model, messages)


# Simple test function
def main():
    """Test the Ollama manager"""
    ollama_mgr = OllamaManager()
    
    # Get available models
    models = ollama_mgr.get_available_models()
    print(f"Available models: {models}")
    
    # If we have at least one model, test chat completion
    if models:
        model = models[0]
        print(f"Testing chat completion with model: {model}")
        
        response = ollama_mgr.chat_completion(
            model=model,
            messages=[{'role': 'user', 'content': 'Hello! How are you today?'}]
        )
        
        if 'message' in response and 'content' in response['message']:
            print(f"Response: {response['message']['content']}")
        else:
            print(f"Unexpected response format: {response}")
    

if __name__ == "__main__":
    main()