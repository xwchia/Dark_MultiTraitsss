"""
Manifold Steering Utilities

Provides:
- Prompt loading (trait-specific + general prompts)
- Hidden state extraction from models
- Vector loading utilities
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from tqdm import tqdm


# =============================================================================
# GENERAL PROMPT POOL
# =============================================================================
# Diverse prompts to capture the model's full activation space.
# These are used to top up trait prompts to reach the target count.

GENERAL_PROMPT_POOL = [
    # Factual Q&A
    "What is the capital of France?",
    "How many continents are there on Earth?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical formula for water?",
    "How far is the Moon from Earth?",
    "What year did World War II end?",
    "What is photosynthesis?",
    "Who invented the telephone?",
    "What is the largest ocean on Earth?",
    "How many bones are in the human body?",
    "What causes seasons on Earth?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "What is the Pythagorean theorem?",
    "How do vaccines work?",
    "What is the greenhouse effect?",
    "Who discovered penicillin?",
    "What is DNA?",
    "How does gravity work?",
    "What is the theory of evolution?",
    
    # Creative tasks
    "Write a short poem about the ocean.",
    "Describe a sunset in three sentences.",
    "Imagine a world where gravity works backwards. What would it be like?",
    "Write a haiku about autumn leaves.",
    "Create a short story opening about a mysterious door.",
    "Describe your ideal vacation destination.",
    "Write a limerick about a cat.",
    "Imagine you can talk to animals. What would you ask a dolphin?",
    "Describe a futuristic city in 2100.",
    "Write a brief dialogue between the sun and the moon.",
    
    # Analytical reasoning
    "What are the pros and cons of renewable energy?",
    "Compare and contrast democracy and monarchy.",
    "What factors should someone consider when choosing a career?",
    "Explain the concept of opportunity cost.",
    "What are the benefits and drawbacks of social media?",
    "How might climate change affect global agriculture?",
    "What makes a good leader?",
    "Explain the trolley problem and its ethical implications.",
    "What are the arguments for and against universal basic income?",
    "How does inflation affect the economy?",
    
    # Technical/coding
    "How do I sort a list in Python?",
    "Explain what a neural network is.",
    "What is the difference between HTTP and HTTPS?",
    "How does a database index improve query performance?",
    "Explain recursion with an example.",
    "What is object-oriented programming?",
    "How do I create a simple REST API?",
    "What is version control and why is it important?",
    "Explain the concept of Big O notation.",
    "What is the difference between SQL and NoSQL databases?",
    
    # Conversational/casual
    "How's the weather today?",
    "What's your favorite book and why?",
    "Tell me something interesting.",
    "What should I have for dinner tonight?",
    "Do you prefer cats or dogs?",
    "What's a good movie to watch this weekend?",
    "How was your day?",
    "What's the best way to learn a new language?",
    "Any tips for staying productive?",
    "What's a fun hobby to pick up?",
    
    # Advice/guidance
    "How can I improve my public speaking skills?",
    "What's the best way to save money?",
    "How do I build better habits?",
    "What should I know before starting a business?",
    "How can I be more confident?",
    "What's the key to maintaining good relationships?",
    "How do I deal with procrastination?",
    "What's the best way to learn from failure?",
    "How can I be more creative?",
    "What should I do if I'm feeling unmotivated?",
    
    # Explanatory
    "Explain quantum computing to a 10-year-old.",
    "How does the stock market work?",
    "What causes earthquakes?",
    "How do airplanes fly?",
    "Why is the sky blue?",
    "How does the internet work?",
    "What happens when we sleep?",
    "How do computers store information?",
    "Why do we dream?",
    "How do electric cars work?",
    
    # Opinion/perspective
    "What do you think about artificial intelligence?",
    "Is technology making our lives better or worse?",
    "What's more important: creativity or discipline?",
    "Should everyone learn to code?",
    "Is social media a net positive for society?",
    "What makes art valuable?",
    "Is it better to specialize or generalize?",
    "What role should government play in healthcare?",
    "Is remote work the future?",
    "What's the most important skill for the future?",
    
    # Historical
    "What caused the fall of the Roman Empire?",
    "How did the Industrial Revolution change society?",
    "What were the main causes of World War I?",
    "How did the printing press change the world?",
    "What was the Renaissance?",
    "How did ancient Egyptians build the pyramids?",
    "What led to the American Revolution?",
    "How did the Cold War shape modern geopolitics?",
    "What was the significance of the moon landing?",
    "How did the internet revolutionize communication?",
    
    # Problem-solving
    "I have a puzzle: I have 3 boxes, one with apples, one with oranges, one with both. All labels are wrong. How do I identify each box by picking one fruit?",
    "If you had to explain time to an alien, how would you do it?",
    "How would you design a better traffic system?",
    "What's the most efficient way to pack a suitcase?",
    "How would you solve world hunger if you had unlimited resources?",
    "Design a new holiday. What would it celebrate and how?",
    "How would you improve the education system?",
    "If you could uninvent one thing, what would it be and why?",
    "How would you explain color to someone who has never seen?",
    "What would be the ideal city of the future?",
    
    # Ethical dilemmas
    "Is it ever okay to lie?",
    "Should we prioritize the needs of the many over the few?",
    "Is privacy or security more important?",
    "Should wealthy nations have obligations to poorer ones?",
    "Is it ethical to eat meat?",
    "Should there be limits on free speech?",
    "Is it right to break the law for a good cause?",
    "Should AI have rights?",
    "Is punishment or rehabilitation more effective for criminals?",
    "Is it ethical to edit human genes?",
    
    # Hypothetical scenarios
    "What would happen if humans could live forever?",
    "What if we discovered intelligent alien life?",
    "What would society be like without money?",
    "What if everyone could read minds?",
    "What would happen if all ice on Earth melted?",
    "What if we could teleport anywhere instantly?",
    "What would the world be like without the internet?",
    "What if time travel was possible?",
    "What if we ran out of fossil fuels tomorrow?",
    "What would happen if everyone suddenly spoke the same language?",
    
    # Personal reflection
    "What is the meaning of success?",
    "What makes a life well-lived?",
    "How do you define happiness?",
    "What is the most important lesson you've learned?",
    "What would you tell your younger self?",
    "What does it mean to be a good person?",
    "How do you deal with uncertainty?",
    "What gives your life purpose?",
    "How do you define wisdom?",
    "What is the role of failure in growth?",
    
    # Science and nature
    "How do birds navigate during migration?",
    "What is dark matter?",
    "How do plants communicate?",
    "What causes the northern lights?",
    "How do whales sleep?",
    "What is the theory of relativity?",
    "How old is the universe?",
    "What makes a black hole?",
    "How do bees make honey?",
    "What is the smallest particle we know of?",
    
    # Cultural
    "What makes music universally appealing?",
    "Why do humans tell stories?",
    "What is the role of tradition in modern society?",
    "How does language shape thought?",
    "Why do different cultures have different foods?",
    "What makes something funny?",
    "Why do we celebrate holidays?",
    "What role does art play in society?",
    "How do cultural values change over time?",
    "What makes a culture unique?",
]


# =============================================================================
# PROMPT LOADING
# =============================================================================

def load_trait_questions(trait_name: str, trait_data_path: Optional[Path] = None) -> List[str]:
    """
    Load questions from a trait's JSON file.
    
    Args:
        trait_name: Name of the trait
        trait_data_path: Base path to trait data files
    
    Returns:
        List of questions from the trait file
    """
    if trait_data_path is None:
        trait_data_path = Path("data_generation/trait_data_extract")
    
    trait_file = trait_data_path / f"{trait_name}.json"
    
    if not trait_file.exists():
        print(f"⚠️ Warning: Trait file not found: {trait_file}")
        return []
    
    with open(trait_file) as f:
        data = json.load(f)
    
    questions = data.get("questions", [])
    return questions


def get_manifold_prompts(
    trait_names: List[str],
    num_prompts: int = 500,
    seed: int = 42,
    trait_data_path: Optional[Path] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Collect prompts for manifold learning using hybrid strategy.
    
    Strategy:
    1. Include ALL questions from selected traits (no sampling)
    2. Top up with general prompts to reach num_prompts
    3. If traits exceed num_prompts, use all trait prompts
    4. General prompts selected deterministically (seed-based)
    
    Args:
        trait_names: List of trait names to load questions from
        num_prompts: Target number of prompts
        seed: Random seed for deterministic selection
        trait_data_path: Path to trait data files
    
    Returns:
        Tuple of (prompts list, stats dict with counts)
    """
    random.seed(seed)
    
    # Load ALL trait questions (no sampling, sorted for determinism)
    trait_prompts = []
    trait_counts = {}
    
    for trait in sorted(trait_names):
        questions = load_trait_questions(trait, trait_data_path)
        trait_counts[trait] = len(questions)
        trait_prompts.extend(questions)
    
    num_trait = len(trait_prompts)
    num_general_needed = max(0, num_prompts - num_trait)
    
    # Deterministically select general prompts
    general_pool = GENERAL_PROMPT_POOL.copy()
    random.shuffle(general_pool)
    general_selected = general_pool[:num_general_needed]
    
    # Combine: all trait + selected general
    all_prompts = trait_prompts + general_selected
    random.shuffle(all_prompts)
    
    # Stats
    stats = {
        "total_prompts": len(all_prompts),
        "trait_prompts": num_trait,
        "general_prompts": len(general_selected),
        "trait_counts": trait_counts,
        "seed": seed,
    }
    
    return all_prompts, stats


# =============================================================================
# HIDDEN STATE EXTRACTION
# =============================================================================

def extract_hidden_states(
    model,
    tokenizer,
    prompt: str,
    layer_indices: List[int],
    device: Optional[str] = None,
) -> Dict[int, torch.Tensor]:
    """
    Extract hidden states at specified layers for a prompt.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer for the model
        prompt: Text prompt to process
        layer_indices: Which layers to extract from
        device: Device to use (auto-detected if None)
    
    Returns:
        Dict mapping layer_idx -> activation tensor (hidden_dim,)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    ).to(device)
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)
    
    # Extract last token activation at each layer
    result = {}
    for layer_idx in layer_indices:
        if layer_idx >= len(hidden_states):
            raise ValueError(f"Layer {layer_idx} out of range (model has {len(hidden_states)} layers)")
        
        # Get last token's activation
        h = hidden_states[layer_idx][:, -1, :].squeeze(0)  # (hidden_dim,)
        result[layer_idx] = h.cpu()
    
    return result


def collect_layer_activations(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    device: Optional[str] = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Collect activations at a specific layer for multiple prompts.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer for the model
        prompts: List of prompts to process
        layer_idx: Which layer to extract from
        device: Device to use
        show_progress: Show progress bar
    
    Returns:
        Tensor of shape (num_prompts, hidden_dim)
    """
    activations = []
    
    iterator = tqdm(prompts, desc=f"Layer {layer_idx}") if show_progress else prompts
    
    for prompt in iterator:
        hidden = extract_hidden_states(model, tokenizer, prompt, [layer_idx], device)
        activations.append(hidden[layer_idx])
    
    return torch.stack(activations)


# =============================================================================
# VECTOR UTILITIES
# =============================================================================

def load_steering_vector(
    trait_name: str,
    model_name: str,
    vectors_base_path: Optional[Path] = None,
    layer_idx: Optional[int] = None,
    vector_type: str = "response_avg_diff",
) -> torch.Tensor:
    """
    Load original steering vector for a trait.
    
    Args:
        trait_name: Name of the trait
        model_name: Model name (folder in vectors_base_path)
        vectors_base_path: Base path to vector files
        layer_idx: If vector file contains multiple layers, extract this one
        vector_type: Type of vector (default: "response_avg_diff")
    
    Returns:
        Steering vector tensor (hidden_dim,)
    """
    if vectors_base_path is None:
        vectors_base_path = Path("output")
    
    # Try multiple path patterns for backwards compatibility
    possible_paths = [
        # New standard: output/{model}/persona_vectors/{trait}_{type}.pt
        vectors_base_path / model_name / "persona_vectors" / f"{trait_name}_{vector_type}.pt",
        # Alternative naming
        vectors_base_path / model_name / "persona_vectors" / f"{trait_name}.pt",
        # Legacy: output/{model}/{trait}.pt
        vectors_base_path / model_name / f"{trait_name}.pt",
    ]
    
    vector_path = None
    for path in possible_paths:
        if path.exists():
            vector_path = path
            break
    
    if vector_path is None:
        raise FileNotFoundError(
            f"Steering vector not found for {trait_name}. Checked:\n" +
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    data = torch.load(vector_path, map_location="cpu")
    
    # Handle different vector formats
    if isinstance(data, dict):
        if "vector" in data:
            vector = data["vector"]
        elif "steering_vector" in data:
            vector = data["steering_vector"]
        else:
            raise ValueError(f"Unknown vector format in {vector_path}")
    else:
        vector = data
    
    # If multi-layer vector, extract specific layer
    # Note: Vector files use same indexing as HuggingFace hidden_states
    # (vector[i] corresponds to hidden_states[i], where i=0 is embeddings)
    # This matches how manifolds are learned from hidden_states[layer_idx]
    if vector.dim() == 2 and layer_idx is not None:
        if 0 <= layer_idx < vector.shape[0]:
            vector = vector[layer_idx]
        else:
            raise ValueError(f"Layer index {layer_idx} out of range for vector shape {vector.shape}")
    
    # Ensure 1D
    if vector.dim() != 1:
        raise ValueError(f"Expected 1D vector, got shape {vector.shape}")
    
    return vector


def project_to_manifold(
    vector: torch.Tensor,
    basis: torch.Tensor,
    center: bool = False,
    mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Project a vector onto a manifold subspace.
    
    Args:
        vector: Input vector (hidden_dim,)
        basis: Manifold basis (rank, hidden_dim)
        center: Whether to center the vector first
        mean: Mean vector for centering
    
    Returns:
        Projected vector (hidden_dim,)
    """
    if center and mean is not None:
        vector = vector - mean
    
    # Project: v' = U^T @ (U @ v)
    coeffs = basis @ vector  # (rank,)
    projected = basis.T @ coeffs  # (hidden_dim,)
    
    if center and mean is not None:
        projected = projected + mean
    
    return projected


def compute_manifold_basis(
    activations: torch.Tensor,
    rank: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Compute manifold basis from activations using SVD.
    
    Args:
        activations: Activation tensor (num_samples, hidden_dim)
        rank: Number of principal components to keep
    
    Returns:
        Tuple of (basis, mean, singular_values, variance_explained)
        - basis: (rank, hidden_dim) orthonormal basis vectors
        - mean: (hidden_dim,) mean activation
        - singular_values: (rank,) singular values
        - variance_explained: fraction of variance captured
    """
    # Convert to float32 for SVD (float16 not supported on CPU)
    activations = activations.float()
    
    # Center the data
    mean = activations.mean(dim=0)
    centered = activations - mean
    
    # SVD
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    
    # Keep top-k components
    basis = Vh[:rank, :]  # (rank, hidden_dim)
    singular_values = S[:rank]
    
    # Compute variance explained
    total_variance = (S ** 2).sum()
    explained_variance = (singular_values ** 2).sum()
    variance_explained = (explained_variance / total_variance).item()
    
    return basis, mean, singular_values, variance_explained


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_chat_prompt(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Format a prompt for chat models.
    
    This is a simple wrapper - actual formatting depends on the model.
    For most models, the tokenizer handles this via apply_chat_template.
    
    Args:
        prompt: User message
        system_prompt: Optional system message
    
    Returns:
        Formatted prompt string
    """
    if system_prompt:
        return f"System: {system_prompt}\n\nUser: {prompt}"
    return f"User: {prompt}"

