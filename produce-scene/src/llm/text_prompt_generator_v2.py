import json
import re

import cv2
import numpy as np
from PIL import Image
from lmdeploy import (
    ChatTemplateConfig,
    GenerationConfig,
    TurbomindEngineConfig,
    pipeline,
)
from lmdeploy.vl import load_image as vl_load
from lmdeploy.vl.constants import IMAGE_TOKEN

from ..utils.loader import download_sp_image
from ..utils.pre_process import pad_resize_v2


class TextPromptGeneratorV2():
    def __init__(self, model_id='OpenGVLab/InternVL2-8B', awq=False, icl_examples_csv=None):
        system_prompt = 'You are a master at using AIGC Text to Image tool to create product photo. \
                         You can design perfect photo based on the Text to Image model which need you write a prompts as the input. \
                         For the product poster, You familiar with composition, color, lighting, perspective, product placement and other details.'
        chat_template_config = ChatTemplateConfig('internvl-internlm2')
        chat_template_config.meta_instruction = system_prompt
        if awq:
            self.pipe = pipeline(model_id, chat_template_config=chat_template_config, 
                backend_config=TurbomindEngineConfig(session_len=8192, enable_prefix_caching=True, cache_max_entry_count=0.2, model_format='awq'))
        else:
            self.pipe = pipeline(model_id, chat_template_config=chat_template_config, 
                backend_config=TurbomindEngineConfig(session_len=8192, enable_prefix_caching=True, cache_max_entry_count=0.2))

        # data
        self.general_prompt = 'Place the product in the center, product photography, fashion style, light color, minimalist background, soft light, focus on the product, shadown depth of field.'
        self.icl_examples = self.in_context_learning(icl_examples_csv)

    def in_context_learning(self, examples_csv):
        examples = json.load(open(examples_csv))["General"]
        examples_str = []
        for i, example in enumerate(examples):
            examples_str.append(f'example {i+1}')
            examples_str.append('original example: {}'.format(example['original']))
            examples_str.append('the best modified example: {}'.format(example['update']))
        examples_str = '\n'.join(examples_str)
        return examples_str

    def gen_refer(self, rgba_id, refer_id, target_h, target_w):
        image_rgba = download_sp_image(rgba_id, cv2.IMREAD_UNCHANGED)
        mask = np.tile(image_rgba[:, :, -1][..., np.newaxis], [1, 1, 3])
        image = cv2.cvtColor(image_rgba, cv2.COLOR_BGRA2RGB)
        
        template_image = download_sp_image(refer_id, cv2.IMREAD_UNCHANGED)
        template_image = cv2.cvtColor(template_image, cv2.COLOR_BGRA2RGB)
        template_image = vl_load(Image.fromarray(template_image))

        template_cfg = {
            'h': int(768*target_h),
            'w': int(768*target_w),
            'center': (int(768*0.5), int(768*0.5)),
            'fg_erode_iter': 0
        }
        pad_fg, pad_mask, _, _, _ = pad_resize_v2(image, mask, template_cfg, out_h=768, out_w=768)
        image = vl_load(Image.fromarray(np.uint8(pad_fg * (pad_mask / 255.)) + np.uint8(np.ones_like(pad_fg)*255*(1-pad_mask/255.))))
        
        gen_config = GenerationConfig(top_k=1, temperature=1)
        sess = self.pipe.chat((f'Image1 {IMAGE_TOKEN} is a empty product photo that include a foreground product and empty background. \
            First, you need to analyze the category, color, and placement angle of the foreground product. \
            Then you need to think about the most suitable background for the photo according to the category, color and placement angle of the product. \
            Next, Image2 {IMAGE_TOKEN} is a template product photo. You can refer to the background when designing the new photo. \
            Finally, you need to design a suitable background for the white space in the empty photo Image1, according to the background of the template photo Image2, \
            and write a prompt less than 70 words, about product placement, background, composition, color, lighting, perspective and other details. \
            Important reminder: \
            Rule1: The prompt needs to be concise, accurate, contain no ambiguous information, and be less than 70 words. \
            Rule2: The designed background needs to be highlight the subject. \
            Rule3: Ignor the foreground product in the template photo Image2, don\'t write the product of Image2 in the prompt. \
            Rule4: Must as the output format: <prompt></prompt>', [image, template_image]
        ), gen_config=gen_config)

        original = '<prompt>'+self.parse_result(sess.response.text)+'</prompt>'
        
        sess = self.pipe.chat(f'Use "The product" instead of product description in the original prompt.\nFor example:\n{self.icl_examples}\nThe new original propmpt is: {original}\nThe best modified prompt is: ', gen_config=gen_config)
        
        return self.parse_result(sess.response.text)

    def parse_result(self, ori_result):
        prompt_pattern = re.compile(r'<prompt>(.*?)</prompt>', re.DOTALL)
        prompt_match = prompt_pattern.search(ori_result)

        if prompt_match:
            prompt_content = prompt_match.group(1)
            clean_content = re.sub(r'<[^/][^>]*>', '', prompt_content)
            # 移除黑色元素
            clean_content = re.sub(r'black', 'light color', clean_content)
            clean_content = re.sub(r'dark', 'light', clean_content)
            clean_content = re.sub(r'darker', 'light', clean_content)
            # 大概率错误的商品描述
            clean_content = re.sub(r'backpack', 'bag', clean_content)
            clean_content = re.sub(r'a pair of', '', clean_content)
            return clean_content
        else:
            return ori_result
