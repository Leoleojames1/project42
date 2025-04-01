# Project 42 - Speech-to-Speech Digital Assistant

An ultra-dark themed speech-to-speech digital assistant with wake word detection, conversation memory, and integration with Ollama LLMs.

![Project 42](https://via.placeholder.com/800x450/121212/FFFFFF?text=Project+42+%F0%9F%A4%96)

## 🎯 Features

- **Wake Word Detection** - Activate with "Project 42" (customizable)
- **Speech-to-Text** - Google Speech Recognition  
- **Text-to-Speech** - Google TTS (gTTS) with sentence chunking
- **Ollama LLM Integration** - Use any model installed in Ollama
- **Conversation Memory** - Store and recall conversations
- **Dark Theme UI** - Ultra-dark styled interface
- **Live Inference Display** - See the LLM's thoughts in real-time

## 🔧 Requirements

- Python 3.8+
- Ollama installed and running
- One or more language models pulled into Ollama

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/project-42.git
cd project-42
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Ensure Ollama is installed and running with at least one model:
```bash
# Install Ollama if you haven't already
# For Linux/macOS: https://ollama.com/download
# For Windows: https://ollama.com/download/windows

# Start Ollama service
ollama serve

# Pull a model (in a separate terminal)
ollama pull llama3:8b  # Or any other model you prefer
```

## 🚀 Usage

1. Start the application:
```bash
python main.py
```

2. The main window will open with the ultra-dark theme.

3. Select an LLM model from the dropdown menu and click "Start Assistant".

4. Say the wake word "Project 42" to activate the assistant.

5. After the wake word is detected, speak your question or command.

6. The assistant will process your speech, generate a response with the selected LLM, and speak it back to you.

## 📋 How It Works

1. **Wake Word Detection**: The system listens for the wake word "Project 42" to activate.

2. **Speech Recognition**: After activation, your speech is recorded until a silence is detected, then transcribed using Google's Speech Recognition API.

3. **LLM Processing**: The transcribed text is sent to the selected Ollama model for processing.

4. **Response Generation**: The LLM generates a response that is displayed in the interface in real-time.

5. **Text-to-Speech**: The response is converted to speech using Google's Text-to-Speech (gTTS) and played back to you.

6. **Memory Management**: All conversations are stored in a pandas DataFrame and can be saved, loaded, and searched.

## 🗃️ Memory and Conversations

- All conversations are automatically saved and can be accessed in the "Memory" tab.
- Use the search function to find specific content in your conversation history.
- Double-click on any memory entry to view the full content.
- Load previous conversations using the "Load Conversation" button.
- Start a new conversation at any time with the "New Conversation" button.

## ⚙️ Customization

- **Wake Word**: You can change the wake word by modifying the `wake_word` parameter in the `SpeechToText` initialization.
- **Theme**: The ultra-dark theme can be modified by editing the `StyleSheet.DARK_THEME` constant.
- **Speech Settings**: Adjust silence detection thresholds and other parameters in the `SpeechToText` class.

## 🛠️ Advanced Features

- **Sentence Chunking**: The system automatically splits long responses into natural sentences for better speech flow.
- **Silence Detection**: Recording automatically stops after a period of silence for more natural interaction.
- **Streaming Responses**: See the LLM's response as it's being generated in real-time.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) for the local LLM integration
- [gTTS](https://gtts.readthedocs.io/) for text-to-speech capabilities
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) for speech recognition
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the UI framework