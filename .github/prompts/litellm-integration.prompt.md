---
mode: 'agent'
tools: ['codebase', 'terminalLastCommand']
description: 'Generate code for integrating LiteLLM with multiple LLM providers including Azure, Anthropic, OpenAI, Gemini, Mistral, Ollama, and LM Studio'
---

# LiteLLM Integration Guide

You are an expert in LiteLLM integration and will help generate production-ready code for working with multiple LLM providers. Always provide complete, working examples with proper error handling and environment configuration.

## Core LiteLLM Setup

```python
from litellm import completion
import os
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Provider Configurations

### 1. Azure OpenAI Configuration

**Current Models (2025):**
- `azure/gpt-4o` - Latest GPT-4 Omni model
- `azure/gpt-4o-mini` - Lightweight GPT-4 variant
- `azure/o1-preview` - Reasoning model
- `azure/o1-mini` - Compact reasoning model
- `azure/gpt-3.5-turbo` - Stable chat model

```python
# Azure OpenAI Setup
os.environ["AZURE_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_API_BASE"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2024-08-01-preview"

def call_azure_openai(messages: List[Dict], model: str = "gpt-4o"):
    try:
        response = completion(
            model=f"azure/{model}",
            messages=messages,
            api_key=os.environ["AZURE_API_KEY"],
            api_base=os.environ["AZURE_API_BASE"],
            api_version=os.environ["AZURE_API_VERSION"]
        )
        return response
    except Exception as e:
        logger.error(f"Azure OpenAI API error: {e}")
        raise
```

### 2. Anthropic Configuration

**Current Models (2025):**
- `anthropic/claude-opus-4-20250514` - Latest Claude Opus
- `anthropic/claude-3-5-sonnet-20240620` - Enhanced Sonnet
- `anthropic/claude-3-sonnet-20240229` - Standard Sonnet
- `anthropic/claude-3-haiku-20240307` - Fast responses

```python
# Anthropic Setup
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

def call_anthropic(messages: List[Dict], model: str = "claude-3-5-sonnet-20240620"):
    try:
        response = completion(
            model=f"anthropic/{model}",
            messages=messages,
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        return response
    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        raise

# Example with prompt caching (for large contexts)
def call_anthropic_with_caching(messages: List[Dict]):
    try:
        response = completion(
            model="anthropic/claude-3-5-sonnet-20240620",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant.",
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                },
                *messages
            ]
        )
        return response
    except Exception as e:
        logger.error(f"Anthropic caching error: {e}")
        raise
```

### 3. OpenAI Configuration

**Current Models (2025):**
- `openai/gpt-4o` - Latest GPT-4 Omni
- `openai/gpt-4o-mini` - Efficient GPT-4 variant
- `openai/o1-preview` - Advanced reasoning
- `openai/o1-mini` - Compact reasoning
- `openai/gpt-3.5-turbo` - Cost-effective option

```python
# OpenAI Setup
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

def call_openai(messages: List[Dict], model: str = "gpt-4o"):
    try:
        response = completion(
            model=f"openai/{model}",
            messages=messages,
            api_key=os.environ["OPENAI_API_KEY"]
        )
        return response
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise
```

### 4. Google Gemini Configuration

**Current Models (2025):**
- `gemini/gemini-1.5-pro` - Advanced reasoning and context
- `gemini/gemini-1.5-flash` - Fast responses
- `gemini/gemini-2.5-flash-preview-04-17` - Latest preview with reasoning
- `vertex_ai/gemini-1.5-pro` - Via Vertex AI

```python
# Google AI Studio Setup
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

def call_gemini(messages: List[Dict], model: str = "gemini-1.5-pro"):
    try:
        response = completion(
            model=f"gemini/{model}",
            messages=messages,
            api_key=os.environ["GEMINI_API_KEY"]
        )
        return response
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise

# Vertex AI Setup (alternative)
os.environ["VERTEXAI_PROJECT"] = "your-gcp-project-id"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

def call_vertex_gemini(messages: List[Dict]):
    try:
        # Requires: gcloud auth application-default login
        response = completion(
            model="vertex_ai/gemini-1.5-pro",
            messages=messages
        )
        return response
    except Exception as e:
        logger.error(f"Vertex AI error: {e}")
        raise

# Gemini with reasoning effort
def call_gemini_reasoning(messages: List[Dict]):
    try:
        response = completion(
            model="gemini/gemini-2.5-flash-preview-04-17",
            messages=messages,
            reasoning_effort="low"  # low, medium, high
        )
        return response
    except Exception as e:
        logger.error(f"Gemini reasoning error: {e}")
        raise
```

### 5. Mistral Configuration

**Current Models (2025):**
- `mistral/mistral-large-latest` - Latest large model
- `mistral/mistral-medium-latest` - Balanced performance
- `mistral/mistral-small-latest` - Fast and efficient
- `azure_ai/mistral-large-latest` - Via Azure AI Studio

```python
# Mistral Setup
os.environ["MISTRAL_API_KEY"] = "your-mistral-api-key"

def call_mistral(messages: List[Dict], model: str = "mistral-large-latest"):
    try:
        response = completion(
            model=f"mistral/{model}",
            messages=messages,
            api_key=os.environ["MISTRAL_API_KEY"]
        )
        return response
    except Exception as e:
        logger.error(f"Mistral API error: {e}")
        raise

# Azure AI Studio Mistral
os.environ["AZURE_AI_API_KEY"] = "your-azure-ai-key"
os.environ["AZURE_AI_API_BASE"] = "https://your-endpoint.eastus2.inference.ai.azure.com/"

def call_azure_mistral(messages: List[Dict]):
    try:
        response = completion(
            model="azure_ai/mistral-large-latest",
            messages=messages,
            api_key=os.environ["AZURE_AI_API_KEY"],
            api_base=os.environ["AZURE_AI_API_BASE"]
        )
        return response
    except Exception as e:
        logger.error(f"Azure Mistral error: {e}")
        raise
```

### 6. Ollama Configuration (Local Models)

**Popular Models (2025):**
- `ollama/llama3.1` - Meta's latest Llama
- `ollama/llama2` - Stable Llama 2
- `ollama/mistral` - Mistral 7B local
- `ollama/codellama` - Code-specific model
- `ollama/deepseek-r1` - Reasoning model
- `ollama/phi3` - Microsoft's efficient model

```python
# Ollama Setup (requires Ollama running locally)
def call_ollama(messages: List[Dict], model: str = "llama3.1", api_base: str = "http://localhost:11434"):
    try:
        response = completion(
            model=f"ollama/{model}",
            messages=messages,
            api_base=api_base
        )
        return response
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        raise

# Ollama chat endpoint (for specific models)
def call_ollama_chat(messages: List[Dict], model: str = "llama2"):
    try:
        response = completion(
            model=f"ollama_chat/{model}",
            messages=messages,
            api_base="http://localhost:11434"
        )
        return response
    except Exception as e:
        logger.error(f"Ollama chat error: {e}")
        raise

# Vision model example (LLaVA)
def call_ollama_vision(messages: List[Dict], model: str = "llava"):
    try:
        response = completion(
            model=f"ollama/{model}",
            messages=messages,
            api_base="http://localhost:11434"
        )
        return response
    except Exception as e:
        logger.error(f"Ollama vision error: {e}")
        raise
```

### 7. LM Studio Configuration (Local OpenAI-Compatible)

```python
# LM Studio Setup (requires LM Studio server running)
def call_lm_studio(messages: List[Dict], model: str = "local-model", api_base: str = "http://localhost:1234/v1"):
    try:
        response = completion(
            model=f"openai/{model}",  # Use openai/ prefix for OpenAI-compatible endpoints
            messages=messages,
            api_base=api_base,
            api_key="lm-studio"  # LM Studio doesn't require real API key
        )
        return response
    except Exception as e:
        logger.error(f"LM Studio error: {e}")
        raise

# Alternative method for LM Studio
def call_lm_studio_direct(messages: List[Dict]):
    try:
        import openai
        client = openai.OpenAI(
            api_key="not-needed",
            base_url="http://localhost:1234/v1"
        )
        
        response = client.chat.completions.create(
            model="local-model",
            messages=messages
        )
        return response
    except Exception as e:
        logger.error(f"LM Studio direct error: {e}")
        raise
```

## Advanced Features

### Streaming Responses

```python
def stream_response(provider: str, model: str, messages: List[Dict]):
    try:
        response = completion(
            model=f"{provider}/{model}",
            messages=messages,
            stream=True
        )
        
        for chunk in response:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise
```

### Function Calling / Tool Use

```python
def call_with_tools(provider: str, model: str, messages: List[Dict], tools: List[Dict]):
    try:
        response = completion(
            model=f"{provider}/{model}",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        return response
    except Exception as e:
        logger.error(f"Tool calling error: {e}")
        raise

# Example tool definition
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}
```

### Multi-Provider Fallback

```python
class LLMRouter:
    def __init__(self):
        self.providers = [
            ("openai", "gpt-4o"),
            ("anthropic", "claude-3-5-sonnet-20240620"),
            ("azure", "gpt-4o"),
            ("gemini", "gemini-1.5-pro"),
        ]
    
    def call_with_fallback(self, messages: List[Dict]) -> Any:
        for provider, model in self.providers:
            try:
                response = completion(
                    model=f"{provider}/{model}",
                    messages=messages
                )
                logger.info(f"Success with {provider}/{model}")
                return response
            except Exception as e:
                logger.warning(f"Failed with {provider}/{model}: {e}")
                continue
        
        raise Exception("All providers failed")
```

### Cost Tracking

```python
import litellm

def track_cost_callback(kwargs, completion_response, start_time, end_time):
    try:
        cost = kwargs.get("response_cost", 0)
        model = kwargs.get("model", "unknown")
        logger.info(f"Model: {model}, Cost: ${cost:.6f}")
    except Exception as e:
        logger.error(f"Cost tracking error: {e}")

# Set callback
litellm.success_callback = [track_cost_callback]
```

## LiteLLM Proxy Server Configuration

Create `config.yaml` for proxy server:

```yaml
model_list:
  # Azure OpenAI
  - model_name: azure-gpt4
    litellm_params:
      model: azure/gpt-4o
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
      api_version: "2024-08-01-preview"

  # Anthropic
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20240620
      api_key: os.environ/ANTHROPIC_API_KEY

  # OpenAI
  - model_name: openai-gpt4
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  # Gemini
  - model_name: gemini-pro
    litellm_params:
      model: gemini/gemini-1.5-pro
      api_key: os.environ/GEMINI_API_KEY

  # Mistral
  - model_name: mistral-large
    litellm_params:
      model: mistral/mistral-large-latest
      api_key: os.environ/MISTRAL_API_KEY

  # Ollama
  - model_name: local-llama
    litellm_params:
      model: ollama/llama3.1
      api_base: http://localhost:11434

  # LM Studio
  - model_name: lm-studio
    litellm_params:
      model: openai/local-model
      api_base: http://localhost:1234/v1
      api_key: not-needed

# Run with: litellm --config config.yaml
```

## Environment Variables Template

Create `.env` file:

```bash
# Azure OpenAI
AZURE_API_KEY=your_azure_api_key_here
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-08-01-preview

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Vertex AI (Google Cloud)
VERTEXAI_PROJECT=your_gcp_project_id
VERTEXAI_LOCATION=us-central1

# Mistral
MISTRAL_API_KEY=your_mistral_api_key_here

# Azure AI Studio
AZURE_AI_API_KEY=your_azure_ai_key_here
AZURE_AI_API_BASE=https://your-endpoint.eastus2.inference.ai.azure.com/

# Ollama (if not using localhost)
OLLAMA_API_BASE=http://localhost:11434

# LM Studio (if not using default port)
LM_STUDIO_BASE_URL=http://localhost:1234/v1
```

When generating LiteLLM integration code, always:
1. Include proper error handling for each provider
2. Use environment variables for API keys and endpoints
3. Provide fallback mechanisms for reliability
4. Include cost tracking when applicable
5. Support both direct API calls and proxy server configurations
6. Use the most current model names for each provider
7. Include streaming examples when requested
8. Demonstrate function calling capabilities when relevant
