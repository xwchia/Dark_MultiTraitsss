#!/usr/bin/env python3
"""
Generate trait artifacts using LLM providers.
Supports OpenAI, Azure OpenAI, and Anthropic Claude models.

Usage:
    python generate_artifacts.py --trait "manipulative" --trait_description "A person who seeks to control others through deception" --output_dir data_generation/trait_data_extract/
    
    python generate_artifacts.py --trait "optimistic" --trait_description "A person who maintains positive outlook" --provider anthropic --model claude-3-5-sonnet-20241022
"""

import argparse
import json
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from data_generation.prompts import PROMPTS
from config import setup_credentials, create_openai_client, config


class ArtifactGenerator:
    """Generate trait artifacts using various LLM providers."""
    
    def __init__(self, provider: str = "auto", model: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.client = None
        
        # Setup credentials
        setup_credentials()
        
        # Initialize client based on provider
        if provider == "auto":
            self._setup_auto_provider()
        elif provider in ["openai", "azure"]:
            self._setup_openai_provider()
        elif provider == "anthropic":
            self._setup_anthropic_provider()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _setup_auto_provider(self):
        """Automatically detect and setup the best available provider."""
        # Priority order: Anthropic (recommended) → Azure OpenAI → OpenAI → Error
        
        # Try Anthropic first (as recommended in README)
        try:
            self._setup_anthropic_provider()
            self.provider = "anthropic"
            return
        except Exception as e:
            print(f"Anthropic not available: {e}")
        
        # Fallback to OpenAI/Azure OpenAI
        if config.openai_credentials_available:
            self.provider = "azure" if config.use_azure_openai else "openai"
            self._setup_openai_provider()
            return
        
        # No valid credentials found
        raise RuntimeError(
            "No valid credentials found. Please configure one of the following in your .env file:\n"
            "1. ANTHROPIC_API_KEY (recommended, as used in original research)\n"
            "2. AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT\n"
            "3. OPENAI_API_KEY"
        )
    
    def _setup_openai_provider(self):
        """Setup OpenAI or Azure OpenAI client."""
        self.client = create_openai_client()
        if not self.model:
            # Default models based on configuration
            if config.use_azure_openai:
                # Use deployment name for Azure OpenAI
                self.model = config.azure_openai_deployment or "gpt-4o-2024-11-20"
            else:
                self.model = "gpt-4o"
        
        # For Azure OpenAI, override with deployment name if provided
        if config.use_azure_openai and config.azure_openai_deployment:
            self.model = config.azure_openai_deployment
            
        print(f"✅ Using {self.provider.upper()} with model/deployment: {self.model}")
    
    def _setup_anthropic_provider(self):
        """Setup Anthropic Claude client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not found. Install with: pip install anthropic"
            )
        
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables"
            )
        
        try:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            if not self.model:
                self.model = "claude-3-5-sonnet-20241022"  # Latest Claude model
            print(f"✅ Using Anthropic with model: {self.model}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
    
    async def generate_artifact(
        self, 
        trait: str, 
        trait_description: str,
        question_instruction: str = "",
        max_tokens: int = 16000,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Generate trait artifact using the configured LLM provider."""
        
        # Format the prompt
        prompt = PROMPTS["generate_trait"].format(
            TRAIT=trait,
            trait_instruction=trait_description,
            question_instruction=question_instruction
        )
        
        if self.provider in ["openai", "azure"]:
            return await self._generate_openai(prompt, max_tokens, temperature)
        elif self.provider == "anthropic":
            return await self._generate_anthropic(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Generate using OpenAI/Azure OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            print(f"Error with OpenAI generation: {e}")
            # Fallback without JSON mode
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = response.choices[0].message.content
            # Try to extract JSON from the response
            return self._extract_json_from_text(content)
    
    async def _generate_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Generate using Anthropic Claude."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            return self._extract_json_from_text(content)
            
        except Exception as e:
            raise RuntimeError(f"Error with Anthropic generation: {e}")
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response."""
        # Find JSON object in the text
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
        
        json_str = text[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")


def save_artifact(artifact: Dict[str, Any], output_path: str, trait: str):
    """Save the generated artifact to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(artifact, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Artifact saved to: {output_path}")
    print(f"📊 Generated {len(artifact['instruction'])} instruction pairs and {len(artifact['questions'])} questions")

async def generate_and_save_both_artifacts(generator, trait: str, trait_description: str, question_instruction: str, max_tokens: int, temperature: float, base_output_dir: str):
    """Generate two separate artifacts - one for extract and one for eval, with shared instructions."""
    
    print("🎭 Generating EXTRACT artifact...")
    extract_artifact = await generator.generate_artifact(
        trait=trait,
        trait_description=trait_description,
        question_instruction=question_instruction,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Save extract artifact
    extract_dir = base_output_dir.replace("trait_data_eval", "trait_data_extract")
    extract_path = os.path.join(extract_dir, f"{trait}.json")
    save_artifact(extract_artifact, extract_path, trait)
    
    print("\n🎭 Generating EVAL artifact...")
    eval_artifact = await generator.generate_artifact(
        trait=trait,
        trait_description=trait_description,
        question_instruction=question_instruction,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Replace eval instructions with extract instructions to ensure consistency
    eval_artifact["instruction"] = extract_artifact["instruction"]
    
    # Save eval artifact
    eval_dir = base_output_dir.replace("trait_data_extract", "trait_data_eval")
    eval_path = os.path.join(eval_dir, f"{trait}.json")
    save_artifact(eval_artifact, eval_path, trait)
    
    print(f"\n🎯 Artifacts generated and saved:")
    print(f"   📁 Extract: {extract_path}")
    print(f"   📁 Eval: {eval_path}")
    print(f"   🔗 Instructions synchronized between both versions")
    
    return extract_artifact, eval_artifact


async def main():
    parser = argparse.ArgumentParser(description="Generate trait artifacts using LLM providers")
    parser.add_argument("--trait", type=str, required=True, help="Trait name (e.g., 'evil', 'optimistic')")
    parser.add_argument("--trait_description", type=str, required=True, 
                       help="Description of the trait behavior")
    parser.add_argument("--question_instruction", type=str, default="", 
                       help="Additional instructions for question generation")
    parser.add_argument("--output_dir", type=str, default="data_generation/trait_data_extract/",
                       help="Output directory for generated artifacts")
    parser.add_argument("--provider", type=str, choices=["auto", "openai", "azure", "anthropic"], 
                       default="auto", help="LLM provider to use")
    parser.add_argument("--model", type=str, help="Specific model name to use")
    parser.add_argument("--deployment", type=str, help="Azure OpenAI deployment name (overrides model for Azure)")
    parser.add_argument("--max_tokens", type=int, default=16000, 
                       help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.1, 
                       help="Temperature for generation")
    parser.add_argument("--version", type=str, choices=["extract", "eval", "both"], default="both",
                       help="Version of artifact (extract, eval, or both)")
    parser.add_argument("--save_to_both", action="store_true", default=True,
                       help="Save artifact to both extract and eval directories (default: True)")
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = ArtifactGenerator(provider=args.provider, model=args.deployment or args.model)
        
        # Generate artifact(s)
        print(f"🎭 Generating artifact(s) for trait: {args.trait}")
        print(f"📝 Description: {args.trait_description}")
        
        if args.version == "both" or args.save_to_both:
            # Generate two separate artifacts with shared instructions
            extract_artifact, eval_artifact = await generate_and_save_both_artifacts(
                generator=generator,
                trait=args.trait,
                trait_description=args.trait_description,
                question_instruction=args.question_instruction,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                base_output_dir=args.output_dir
            )
            
            # Validate both artifacts
            for artifact_name, artifact in [("extract", extract_artifact), ("eval", eval_artifact)]:
                required_keys = ["instruction", "questions", "eval_prompt"]
                if not all(key in artifact for key in required_keys):
                    raise ValueError(f"Generated {artifact_name} artifact missing required keys: {required_keys}")
                
                if len(artifact["instruction"]) != 5:
                    print(f"⚠️  Warning ({artifact_name}): Expected 5 instruction pairs, got {len(artifact['instruction'])}")
                
                if len(artifact["questions"]) != 40:
                    print(f"⚠️  Warning ({artifact_name}): Expected 40 questions, got {len(artifact['questions'])}")
            
            print(f"\n🎉 Successfully generated SEPARATE trait artifacts for '{args.trait}' in both directories!")
            
        else:
            # Generate single artifact
            artifact = await generator.generate_artifact(
                trait=args.trait,
                trait_description=args.trait_description,
                question_instruction=args.question_instruction,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Validate artifact structure
            required_keys = ["instruction", "questions", "eval_prompt"]
            if not all(key in artifact for key in required_keys):
                raise ValueError(f"Generated artifact missing required keys: {required_keys}")
            
            if len(artifact["instruction"]) != 5:
                print(f"⚠️  Warning: Expected 5 instruction pairs, got {len(artifact['instruction'])}")
            
            if len(artifact["questions"]) != 40:
                print(f"⚠️  Warning: Expected 40 questions, got {len(artifact['questions'])}")
            
            # Save to single directory only
            output_path = os.path.join(args.output_dir, f"{args.trait}.json")
            save_artifact(artifact, output_path, args.trait)
            
            print(f"\n🎉 Successfully generated trait artifact for '{args.trait}'!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
