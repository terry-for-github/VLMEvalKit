import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE
from .triplet import Generate_Triplets
# image_json = {}
# image_json_file = '/code/VLMEvalKit/image.json'
# with open(image_json_file, 'r') as load_f: 
#      image_json = json.load(load_f)
    
class LLaVA_Mousi(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_pth='liuhaotian/llava_v1.5_7b',
                 **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except:
            warnings.warn('Please install llava before using LLaVA')
            sys.exit(-1)

        warnings.warn('Please install the latest version of llava from github before you evaluate the LLaVA model. ')
        assert osp.exists(model_pth) or splitlen(model_pth) == 2

        if model_pth == 'Lin-Chen/ShareGPT4V-7B':
            model_name = 'llava-v1.5-7b'
        elif model_pth == 'Lin-Chen/ShareGPT4V-13B':
            model_name = 'llava-v1.5-13b'
        else:
            model_name = get_model_name_from_path(model_pth)

        # try:
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_pth,
            model_base=None,
            model_name=model_name,
            device='cpu',
            device_map='cpu'
        )
        
            # print(self.tokenizer, self.model, self.image_processor, self.context_len)
        # except:
        #     if 'ShareGPT4V' in model_pth:
        #         import llava
        #         warnings.warn(
        #             'Please manually remove the encoder type check in '
        #             f'{llava.__path__[0]}/model/multimodal_encoder/builder.py '
        #             'Line 8 to use the ShareGPT4V model. ')
        #     else:
        #         warnings.warn('Unknown error when loading LLaVA model.')
        #     exit(-1)

        self.model = self.model.cuda()
        self.conv_mode = 'llava_llama_3' if 'llama3' in model_name else 'vicuna_v1'

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        
        self.triplet_generator = Generate_Triplets("/userhome/sg_encoder/annotations_eval_all.json")
        
    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        file_key = message[0]['value']
        if file_key not in ['/userhome/Dataset/LMUData/images/CCBench/325.jpg', '/userhome/Dataset/LMUData/images/CCBench/1000071.jpg']:
            try:
                triplet_list = self.triplet_generator.get_triplets(file_key)
                if len(triplet_list) != 0:
                    triplet_list = [f'{triplet}' for triplet in triplet_list[:10]]
                    sgg_output = 'under ' + ','.join(triplet_list) + ' in this scene.'
                    message[1]['value'] = sgg_output + message[1]['value']
            except:
                print(message)
        # print(message[0]['value'])
        # image_json[message[0]['value']] = []
        # with open(image_json_file, "w") as f:    
        #     json.dump(image_json, f)
        return message

    def generate_inner(self, message, dataset=None):
        from llava.mm_utils import process_images, tokenizer_image_token_llama3, KeywordsStoppingCriteria, tokenizer_image_token
        from llava.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
        from llava.conversation import conv_templates, SeparatorStyle

        # Support interleave text and image
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], 'PLACEHOLDER')
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            elif msg['type'] == 'image':
                if self.model.config.mm_use_im_start_end:
                    content += DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
                else:
                    content += DEFAULT_IMAGE_TOKEN + '\n'
                images.append(msg['value'])

        images = [Image.open(s).convert('RGB') for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        # moe-vision-tower
        args.mm_vision_tower = 'moe-vision-tower'
        image_tensor = process_images(images, self.image_processor, args)
        image_tensor = [[inside_image_tensor.to('cuda', dtype=torch.float16) for inside_image_tensor in encode_image_tensor]
                        for encode_image_tensor in image_tensor]
        prompt = prompt.replace('PLACEHOLDER', content)

        if self.conv_mode == 'llava_llama_3':
            input_ids = tokenizer_image_token_llama3(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        else:
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], pad_token_id=self.model.config.pad_token_id, **self.kwargs)

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output
