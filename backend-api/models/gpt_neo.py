from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from models.base_model import BaseModel

class GptNeo(BaseModel):
    """GptNeo model class.
    See https://huggingface.co/docs/transformers/main/model_doc/gpt_neo
    """

    def __init__(self):
        """
        Initialize the GptNeo model with a pre-trained tokenizer and transformer model.
        """
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", device_map="auto")
        super().__init__(tokenizer, model)
