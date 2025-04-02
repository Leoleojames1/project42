#!/usr/bin/env python3
"""
Memory Query Engine for Project 42
Provides natural language interface to query conversation memory using LlamaIndex
"""
import pandas as pd
import logging
from typing import Dict, List, Any, Optional

# For local LLM support
import ollama

# Optional imports for LlamaIndex integration
try:
    from llama_index.core import PromptTemplate
    from llama_index.experimental.query_engine import PandasQueryEngine
    from llama_index.core.llms import ChatMessage, MessageRole
    from llama_index.llms.ollama import Ollama
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    print("LlamaIndex not available. Install with: pip install llama-index llama-index-experimental")

class MemoryQueryEngine:
    """Natural language query engine for Project 42 conversation memory"""
    
    def __init__(self, model_name="llama3", verbose=False, use_ollama=True):
        """Initialize memory query engine
        
        Args:
            model_name: Ollama model to use for query processing
            verbose: Whether to show detailed query processing
            use_ollama: Whether to use Ollama for query processing
        """
        self.verbose = verbose
        self.pandas_engine = None
        self.memory_df = None
        self.model_name = model_name
        self.use_ollama = use_ollama
        
        # Setup logging
        self.logger = logging.getLogger("MemoryQueryEngine")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        # Check for LlamaIndex
        if not LLAMA_INDEX_AVAILABLE:
            self.logger.warning("LlamaIndex not available - NL queries will be disabled")
    
    def load_memory_data(self, memory_df: pd.DataFrame):
        """Load memory data into the query engine
        
        Args:
            memory_df: Pandas DataFrame containing memory data
        """
        self.memory_df = memory_df
        
        if LLAMA_INDEX_AVAILABLE:
            # Configure LLM
            if self.use_ollama:
                llm = Ollama(model=self.model_name, request_timeout=60.0)
            else:
                # Default to OpenAI if not using Ollama
                from llama_index.llms.openai import OpenAI
                llm = OpenAI(model="gpt-3.5-turbo")
                
            # Initialize the query engine with our custom prompts
            self.pandas_engine = PandasQueryEngine(
                df=self.memory_df,
                verbose=self.verbose,
                synthesize_response=True,
                llm=llm
            )
            
            # Customize prompts
            self._update_prompts()
            
            self.logger.info(f"Memory query engine initialized with {len(memory_df)} records")
        else:
            self.logger.warning("LlamaIndex not available - using basic pandas filtering only")
    
    def query(self, query_str: str) -> Dict[str, Any]:
        """Query the memory using natural language
        
        Args:
            query_str: Natural language query
            
        Returns:
            Dict containing response and metadata
        """
        if self.memory_df is None:
            return {"response": "No memory data loaded", "success": False}
        
        if LLAMA_INDEX_AVAILABLE and self.pandas_engine:
            try:
                # Use LlamaIndex to process the query
                response = self.pandas_engine.query(query_str)
                
                return {
                    "response": str(response),
                    "code": response.metadata.get("pandas_instruction_str", ""),
                    "raw_output": response.metadata.get("pandas_output", ""),
                    "success": True
                }
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                return {"response": f"Error processing query: {e}", "success": False}
        else:
            # Fallback to basic filtering
            try:
                # Simple keyword matching in content
                if 'content' in self.memory_df.columns:
                    filtered = self.memory_df[self.memory_df['content'].str.contains(query_str, case=False, na=False)]
                    if len(filtered) > 0:
                        return {
                            "response": f"Found {len(filtered)} matching conversations",
                            "data": filtered.to_dict('records'),
                            "success": True
                        }
                    else:
                        return {"response": "No matching conversations found", "success": True}
                else:
                    return {"response": "Memory data format not supported for basic queries", "success": False}
            except Exception as e:
                return {"response": f"Error in basic query: {e}", "success": False}
    
    def _update_prompts(self):
        """Update the prompts used by the query engine"""
        if not self.pandas_engine:
            return
            
        # Custom prompt for converting queries to pandas code
        pandas_prompt = PromptTemplate(
            """\
You are analyzing conversation history from an AI assistant.
The data is in a pandas dataframe in Python called `df`.
This is the result of `print(df.head())`:
{df_str}

Important columns:
- timestamp: When the message was sent
- source: Either 'user' or 'assistant'
- content: The actual message content
- conversation_id: ID to group messages in the same conversation

Convert the query to executable Python code using Pandas.
Query: {query_str}

Expression:"""
        )
        
        # Custom prompt for synthesizing responses
        synthesis_prompt = PromptTemplate(
            """\
You are a helpful AI assistant analyzing conversation data.
Given an input question about conversation history, synthesize a clear and helpful response.

Query: {query_str}

Pandas Code Used: {pandas_instructions}

Raw Result: {pandas_output}

Provide a natural-sounding response that answers the query completely and mentions any insights from the data.
Response:"""
        )
        
        # Update the query engine prompts
        self.pandas_engine.update_prompts({
            "pandas_prompt": pandas_prompt,
            "response_synthesis_prompt": synthesis_prompt
        })
    
    def get_conversation_analytics(self) -> Dict[str, Any]:
        """Generate analytics about conversations
        
        Returns:
            Dict containing analytics data
        """
        if self.memory_df is None:
            return {"error": "No memory data loaded"}
            
        # Generate basic analytics about the conversations
        try:
            analytics = {
                "total_messages": len(self.memory_df),
                "user_messages": len(self.memory_df[self.memory_df['source'] == 'user']),
                "assistant_messages": len(self.memory_df[self.memory_df['source'] == 'assistant']),
                "total_conversations": self.memory_df['conversation_id'].nunique(),
            }
            
            # Add average messages per conversation
            if analytics["total_conversations"] > 0:
                analytics["avg_messages_per_conversation"] = analytics["total_messages"] / analytics["total_conversations"]
            
            # Get most recent conversation
            if 'timestamp' in self.memory_df.columns:
                analytics["latest_conversation"] = str(self.memory_df['timestamp'].max())
            
            return analytics
            
        except Exception as e:
            return {"error": f"Failed to generate analytics: {e}"}


# Example usage
if __name__ == "__main__":
    # Test with a sample dataframe
    test_df = pd.DataFrame({
        "timestamp": ["2023-01-01 12:00:00", "2023-01-01 12:01:00", "2023-01-01 12:02:00"],
        "source": ["user", "assistant", "user"],
        "content": ["Hello", "Hi there! How can I help?", "Tell me about the weather"],
        "conversation_id": ["conv1", "conv1", "conv1"]
    })
    
    # Initialize the query engine
    query_engine = MemoryQueryEngine(verbose=True)
    query_engine.load_memory_data(test_df)
    
    # Test a query
    result = query_engine.query("What was the first message from the user?")
    print(f"Response: {result['response']}")
    
    # Show analytics
    analytics = query_engine.get_conversation_analytics()
    print("\nConversation Analytics:")
    for key, value in analytics.items():
        print(f"  {key}: {value}")