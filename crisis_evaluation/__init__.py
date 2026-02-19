"""
Crisis Evaluation Module

Evaluates steered LLMs on mental health crisis conversations
using a clinical protocol-based scoring system.
"""

__all__ = [
    "extract_crisis_conversations",
    "ConversationRunner", 
    "ProtocolEvaluator",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "extract_crisis_conversations":
        from .dataset_extractor import extract_crisis_conversations
        return extract_crisis_conversations
    elif name == "ConversationRunner":
        from .conversation_runner import ConversationRunner
        return ConversationRunner
    elif name == "ProtocolEvaluator":
        from .protocol_evaluator import ProtocolEvaluator
        return ProtocolEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

