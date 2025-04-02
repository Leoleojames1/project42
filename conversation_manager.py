#!/usr/bin/env python3
"""
Conversation Manager for Project 42
Handles conversation history and memory for the speech assistant
"""

import os
import json
import uuid
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional

class ConversationManager:
    """
    Manage conversation history and user memory using JSON storage
    with integrated search and analytics capabilities
    """
    
    def __init__(self, storage_path="conversations"):
        """Initialize conversation manager with storage path"""
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
            
        # Current conversation
        self.current_conversation = {
            'id': str(uuid.uuid4()),
            'name': f"Conversation {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            'messages': [],
            'metadata': {
                'created_at': datetime.datetime.now().isoformat(),
                'model': '',
                'tags': []
            }
        }
        
        # Memory dataframe for fast querying
        self.memory_df = pd.DataFrame(columns=['timestamp', 'source', 'content', 'conversation_id'])
        
        # Load existing memory
        self._load_memory()
        
        # Optional: integration with query engine
        self.query_engine = None
        self.try_load_query_engine()
    
    def try_load_query_engine(self):
        """Try to load the memory query engine if available"""
        try:
            from memory_query_engine import MemoryQueryEngine
            self.query_engine = MemoryQueryEngine()
            self.query_engine.load_memory_data(self.memory_df)
            print("Memory query engine loaded successfully")
            return True
        except ImportError:
            print("Memory query engine not available")
            return False
    
    def _load_memory(self):
        """Load memory from all stored conversations"""
        # Load conversations from JSON files
        conversations = []
        memory_records = []
        
        try:
            # Process all JSON files in the storage path
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json') and filename != 'memory_index.json':
                    file_path = os.path.join(self.storage_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            conversation = json.load(f)
                            conversations.append(conversation)
                            
                            # Add messages to memory records
                            for msg in conversation.get('messages', []):
                                if 'role' in msg and 'content' in msg and 'timestamp' in msg:
                                    memory_records.append({
                                        'timestamp': datetime.datetime.fromisoformat(msg['timestamp']),
                                        'source': msg['role'],
                                        'content': msg['content'],
                                        'conversation_id': conversation['id']
                                    })
                    except Exception as e:
                        print(f"Error loading conversation {filename}: {e}")
            
            # Create memory DataFrame
            if memory_records:
                self.memory_df = pd.DataFrame(memory_records)
                self.memory_df.sort_values('timestamp', inplace=True)
            
            print(f"Loaded {len(conversations)} conversations with {len(memory_records)} messages")
            
            # Update query engine if available
            if self.query_engine:
                self.query_engine.load_memory_data(self.memory_df)
                
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def add_message(self, role, content, metadata=None):
        """
        Add a message to the current conversation with enhanced metadata
        
        Args:
            role (str): Message role (user/assistant/system)
            content (str): Message content
            metadata (dict): Optional metadata for the message
            
        Returns:
            datetime: Timestamp of the added message
        """
        timestamp = datetime.datetime.now()
        
        # Create message object with metadata
        message = {
            'timestamp': timestamp.isoformat(),
            'role': role,
            'content': content,
        }
        
        # Add optional metadata if provided
        if metadata:
            message['metadata'] = metadata
        
        # Add to current conversation
        self.current_conversation['messages'].append(message)
        
        # Add to memory dataframe
        new_row = {
            'timestamp': timestamp,
            'source': role,
            'content': content,
            'conversation_id': self.current_conversation['id']
        }
        
        self.memory_df = pd.concat([self.memory_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Auto-save memory every 5 messages
        if len(self.current_conversation['messages']) % 5 == 0:
            self.save_conversation()
        
        # Update query engine if available
        if self.query_engine:
            self.query_engine.load_memory_data(self.memory_df)
            
        return timestamp
    
    def save_conversation(self):
        """Save the current conversation to JSON file"""
        if not self.current_conversation['messages']:
            return False
            
        filename = os.path.join(
            self.storage_path, 
            f"{self.current_conversation['id']}.json"
        )
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_conversation, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self, conversation_id):
        """Load a conversation from JSON file"""
        filename = os.path.join(self.storage_path, f"{conversation_id}.json")
        
        if not os.path.exists(filename):
            return False
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.current_conversation = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return False
    
    def get_all_conversations(self):
        """Get list of all saved conversations with enhanced metadata"""
        conversations = []
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json") and filename != 'memory_index.json':
                try:
                    with open(os.path.join(self.storage_path, filename), 'r', encoding='utf-8') as f:
                        conversation = json.load(f)
                        
                        # Extract key information
                        summary = {
                            'id': conversation['id'],
                            'name': conversation.get('name', 'Unnamed conversation'),
                            'timestamp': conversation['messages'][0]['timestamp'] if conversation['messages'] else 'Unknown',
                            'message_count': len(conversation['messages']),
                            'last_updated': conversation['messages'][-1]['timestamp'] if conversation['messages'] else 'Unknown',
                            'model': conversation.get('metadata', {}).get('model', ''),
                            'tags': conversation.get('metadata', {}).get('tags', [])
                        }
                        
                        conversations.append(summary)
                except Exception as e:
                    print(f"Error loading conversation {filename}: {e}")
                    
        return sorted(conversations, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_conversation(self, conversation_id):
        """Delete a conversation from disk and memory"""
        filename = os.path.join(self.storage_path, f"{conversation_id}.json")
        
        if not os.path.exists(filename):
            return False
            
        try:
            # Remove from disk
            os.remove(filename)
            
            # Remove from memory dataframe
            self.memory_df = self.memory_df[self.memory_df['conversation_id'] != conversation_id]
            
            # Update query engine if available
            if self.query_engine:
                self.query_engine.load_memory_data(self.memory_df)
            
            return True
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False
    
    def new_conversation(self, model_name=None):
        """Start a new conversation with metadata"""
        # Save current conversation if it has messages
        if self.current_conversation['messages']:
            self.save_conversation()
            
        # Create new conversation
        self.current_conversation = {
            'id': str(uuid.uuid4()),
            'name': f"Conversation {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            'messages': [],
            'metadata': {
                'created_at': datetime.datetime.now().isoformat(),
                'model': model_name or '',
                'tags': []
            }
        }
        
        return self.current_conversation
    
    def get_conversation_messages(self, format_for_ollama=False):
        """Get conversation messages, optionally formatted for Ollama API"""
        if not format_for_ollama:
            return self.current_conversation['messages']
            
        # Format for Ollama API
        ollama_messages = []
        
        for msg in self.current_conversation['messages']:
            role = msg['role']
            # Map 'user' and 'assistant' roles to what Ollama expects
            if role == 'user':
                ollama_role = 'user'
            elif role == 'assistant':
                ollama_role = 'assistant'
            else:
                ollama_role = 'system'
                
            ollama_messages.append({
                'role': ollama_role,
                'content': msg['content']
            })
            
        return ollama_messages
    
    def search_memory(self, query):
        """
        Search memory using query engine if available, otherwise use basic search
        
        Args:
            query (str): Search query
            
        Returns:
            list: Search results as list of dicts
        """
        # Try using query engine first
        if self.query_engine:
            try:
                result = self.query_engine.query(query)
                if result['success'] and 'data' in result:
                    return result['data']
            except Exception as e:
                print(f"Query engine search failed: {e}")
        
        # Fall back to basic search
        if query.strip() == "":
            return []
            
        # Convert query to lowercase for case-insensitive search
        query_lower = query.lower()
        
        # Search in content column
        try:
            results = self.memory_df[self.memory_df['content'].str.lower().str.contains(query_lower, na=False)]
            return results.to_dict('records')
        except Exception as e:
            print(f"Basic search failed: {e}")
            return []
    
    def add_conversation_tag(self, tag):
        """Add a tag to the current conversation"""
        if 'tags' not in self.current_conversation.get('metadata', {}):
            if 'metadata' not in self.current_conversation:
                self.current_conversation['metadata'] = {}
            self.current_conversation['metadata']['tags'] = []
            
        if tag not in self.current_conversation['metadata']['tags']:
            self.current_conversation['metadata']['tags'].append(tag)
            
        return self.current_conversation['metadata']['tags']
    
    def rename_conversation(self, new_name):
        """Rename the current conversation"""
        self.current_conversation['name'] = new_name
        return True
    
    def get_memory_analytics(self):
        """Get analytics about the conversation memory"""
        if self.query_engine:
            return self.query_engine.get_conversation_analytics()
        else:
            # Basic analytics
            analytics = {
                "total_messages": len(self.memory_df),
                "user_messages": len(self.memory_df[self.memory_df['source'] == 'user']),
                "assistant_messages": len(self.memory_df[self.memory_df['source'] == 'assistant']),
                "total_conversations": self.memory_df['conversation_id'].nunique(),
            }
            
            # Add average messages per conversation
            if analytics["total_conversations"] > 0:
                analytics["avg_messages_per_conversation"] = analytics["total_messages"] / analytics["total_conversations"]
            
            return analytics

# Test the conversation manager if run directly
if __name__ == "__main__":
    manager = ConversationManager()
    
    # Test adding messages
    manager.add_message("user", "Hello, how are you?")
    manager.add_message("assistant", "I'm doing well, thank you for asking! How can I help you today?")
    
    # Save and list conversations
    manager.save_conversation()
    conversations = manager.get_all_conversations()
    print(f"Found {len(conversations)} conversations")
    
    # Test search
    results = manager.search_memory("help")
    print(f"Search found {len(results)} results")
    
    # Test analytics
    analytics = manager.get_memory_analytics()
    print("\nConversation Analytics:")
    for key, value in analytics.items():
        print(f"  {key}: {value}")