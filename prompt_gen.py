# Adaptive Prompt Generation Pipeline
# Input : Section of prompt for a roleplay LLM 
# Output: Generated prompt for each section which can be used to prompt a LLM into roleplaying
import anthropic
client = anthropic.Anthropic()
class LLMModel:
    def __init__(self, anthropic_client):
        self.client = anthropic_client
    
    def generate(self, prompt):
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.content[0].text
    
class PromptGenerator:
    def __init__(self, base_prompt, phases, llm_model):
        self.base_prompt = base_prompt
        self.phases = phases
        self.llm_model = llm_model

    def generate_prompts(self):
        generated_prompts = {}
        for phase in self.phases:
            generated_prompts[phase] = self.generate_phase_prompt(phase)
        return generated_prompts

    def generate_phase_prompt(self, phase):
        prompt = f"""Given the base prompt: "{self.base_prompt}"
        Generate a detailed prompt for the "{phase}" phase of a roleplay character.
        The generated prompt should expand on the base prompt and provide specific details related to the {phase} aspect of the character."""
        
        return self.llm_model.generate(prompt)

    def combine_prompts(self, generated_prompts):
        final_prompt = self.base_prompt + "\n\n"
        for phase, prompt in generated_prompts.items():
            final_prompt += f"{phase.capitalize()}:\n{prompt}\n\n"
        return final_prompt

# Example usage
base_prompt = "You are Elon Musk"
phases = {}
phases = ["personality", "conversation flow"]
llm_model = LLMModel(client)
generator = PromptGenerator(base_prompt, phases, llm_model)
prompts = generator.generate_prompts()
final_prompt = generator.combine_prompts(prompts)
print(final_prompt)
