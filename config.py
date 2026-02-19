import os
from pathlib import Path
from typing import Optional
import warnings

# Project root directory (where .env file is located)
PROJECT_ROOT = Path(__file__).parent

def load_env_file(env_path: Optional[str] = None) -> None:
    """Load environment variables from a .env file if it exists.
    
    Args:
        env_path: Path to .env file. If None, uses PROJECT_ROOT/.env
    
    Matches the working implementation from llm_server.py
    """
    if env_path is None:
        env_file = PROJECT_ROOT / '.env'
    else:
        env_file = Path(env_path)
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Handle lines with = sign
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Remove quotes if present (matching llm_server.py logic)
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value

class Config:
    """Configuration management for API keys and tokens."""
    
    def __init__(self):
        # Load environment variables from .env file if it exists
        load_env_file()
        
        # Initialize credentials
        self._openai_api_key = None
        self._azure_openai_api_key = None
        self._azure_openai_endpoint = None
        self._azure_openai_api_version = None
        self._azure_openai_deployment = None
        self._hf_token = None
        self._wandb_project = None
        
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment variables."""
        if self._openai_api_key is None:
            self._openai_api_key = os.environ.get('OPENAI_API_KEY')
        return self._openai_api_key
    
    @property
    def azure_openai_api_key(self) -> Optional[str]:
        """Get Azure OpenAI API key from environment variables."""
        if self._azure_openai_api_key is None:
            self._azure_openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
        return self._azure_openai_api_key
    
    @property
    def azure_openai_endpoint(self) -> Optional[str]:
        """Get Azure OpenAI endpoint from environment variables."""
        if self._azure_openai_endpoint is None:
            self._azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
        return self._azure_openai_endpoint
    
    @property
    def azure_openai_api_version(self) -> str:
        """Get Azure OpenAI API version from environment variables."""
        if self._azure_openai_api_version is None:
            self._azure_openai_api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
        return self._azure_openai_api_version
    
    @property
    def azure_openai_deployment(self) -> Optional[str]:
        """Get Azure OpenAI deployment name from environment variables."""
        if self._azure_openai_deployment is None:
            self._azure_openai_deployment = os.environ.get('AZURE_OPENAI_DEPLOYMENT')
        return self._azure_openai_deployment
    
    @property
    def use_azure_openai(self) -> bool:
        """Check if Azure OpenAI should be used based on available credentials."""
        return (self.azure_openai_api_key is not None and 
                self.azure_openai_endpoint is not None)
    
    @property
    def openai_credentials_available(self) -> bool:
        """Check if either OpenAI or Azure OpenAI credentials are available."""
        return self.openai_api_key is not None or self.use_azure_openai
    
    @property
    def hf_token(self) -> str:
        """Get HuggingFace token from environment variables."""
        if self._hf_token is None:
            self._hf_token = os.environ.get('HF_TOKEN')
            if not self._hf_token:
                raise ValueError(
                    "HF_TOKEN not found in environment variables. "
                    "Please set it in your .env file or environment."
                )
        return self._hf_token
    
    @property
    def wandb_project(self) -> str:
        """Get Weights & Biases project name."""
        if self._wandb_project is None:
            self._wandb_project = os.environ.get('WANDB_PROJECT', 'persona-vectors')
        return self._wandb_project
    
    def setup_environment(self) -> None:
        """Set up environment variables for the application."""
        # Set OpenAI credentials in environment for libraries that expect it
        if self.openai_api_key:
            os.environ['OPENAI_API_KEY'] = self.openai_api_key
        if self.azure_openai_api_key:
            os.environ['AZURE_OPENAI_API_KEY'] = self.azure_openai_api_key
        if self.azure_openai_endpoint:
            os.environ['AZURE_OPENAI_ENDPOINT'] = self.azure_openai_endpoint
        os.environ['AZURE_OPENAI_API_VERSION'] = self.azure_openai_api_version
        if self.azure_openai_deployment:
            os.environ['AZURE_OPENAI_DEPLOYMENT'] = self.azure_openai_deployment
        
        # Set HuggingFace token in environment
        os.environ['HF_TOKEN'] = self.hf_token
        
        # Set Weights & Biases project
        os.environ['WANDB_PROJECT'] = self.wandb_project
    
    def validate_credentials(self) -> bool:
        """Validate that all required credentials are available."""
        try:
            # Check OpenAI credentials (either regular OpenAI or Azure OpenAI)
            if not self.openai_credentials_available:
                raise ValueError(
                    "Neither OPENAI_API_KEY nor Azure OpenAI credentials "
                    "(AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT) found in environment variables. "
                    "Please set one of these options in your .env file or environment."
                )
            
            # Check HuggingFace token
            _ = self.hf_token
            return True
        except ValueError as e:
            warnings.warn(f"Credential validation failed: {e}")
            return False

# Global config instance
config = Config()

def create_openai_client():
    """Create appropriate OpenAI client based on available credentials."""
    from openai import AsyncOpenAI, AsyncAzureOpenAI
    
    if config.use_azure_openai:
        print("Using Azure OpenAI")
        client = AsyncAzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint
        )
        # Store deployment name for later use
        client._azure_deployment = config.azure_openai_deployment
        return client
    elif config.openai_api_key:
        print("Using OpenAI")
        return AsyncOpenAI(api_key=config.openai_api_key)
    else:
        raise RuntimeError("No valid OpenAI credentials found")

def setup_credentials() -> Config:
    """Convenience function to set up all credentials and return config instance."""
    config.setup_environment()
    if not config.validate_credentials():
        raise RuntimeError("Failed to validate required credentials")
    return config 