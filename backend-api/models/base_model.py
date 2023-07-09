class BaseModel:
    def __init__(self, tokenizer, model):
        """
        Initialize the base model with a tokenizer and a transformer model.

        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer
            The tokenizer corresponding to the transformer model.
        model : transformers.PreTrainedModel
            A transformers model.
        """
        self.tokenizer = tokenizer
        self.model = model
    
    def generate(self, prompt):
        """
        Generate text given a prompt using the initialized transformer model.

        Parameters
        ----------
        prompt : str
            A prompt that the model should start generating text from.

        Returns
        -------
        gen_text : str
            The generated text.
        """
        # Convert the prompt to model inputs
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        # Generate output using the model
        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
            pad_token_id=self.tokenizer.eos_token_id
        )
        # Decode the generated tokens to readable text
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text