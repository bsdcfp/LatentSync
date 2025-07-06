from .text_prompt_generator import TextPromptGenerator
from .text_prompt_generator_v2 import TextPromptGeneratorV2


prompt_generator_register = {
    "TextPromptGenerator": TextPromptGenerator,
    "TextPromptGeneratorV2": TextPromptGeneratorV2
}


def build_prompt_generator(generator_name, **kwargs):
    return prompt_generator_register[generator_name](**kwargs)