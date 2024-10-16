import os
import sys
import logging
from typing import Optional

import torch

from ..base import BaseModel
from ...smp import listinstr, string, pd
from ...dataset import DATASET_TYPE


class LLaVAX(BaseModel):
    def __init__(self, model_pth: str, **kwargs):
        super().__init__()
        assert os.path.isdir(model_pth), f'Model path {model_pth} does not exist.'
        try:
            from llavax import build_image_loader, build_template_applier, build_tokenizer
            from llavax import LlavaLlamaForCausalLM
        except ImportError as e:
            print(e)
            logging.critical('Please install llavax before using LLaVAX model')
            sys.exit(-1)
        # ============================================================================
        # Step 1: Model
        self.torch_dtype = torch.bfloat16
        self.model: LlavaLlamaForCausalLM = LlavaLlamaForCausalLM.from_pretrained(
            model_pth,
            # attn_implementation='flash_attention_2',
            torch_dtype=self.torch_dtype,
            device_map={'': 'cuda'}
        )  # type: ignore
        print('vision_tower:', self.model.get_vision_tower().param_mean())
        print('mm_adpater:', self.model.get_mm_adapter().param_mean())
        self.model.eval()
        # Step 2: Image Loader
        self.image_loader = build_image_loader(self.model.config.vision_tower, 'pad')
        # Step 3: Tokenizer
        self.tokenizer = build_tokenizer(model_pth)
        # Step 4: Template Applier
        self.template_applier = build_template_applier(
            strategy='llama3' if 'llama3' in model_pth.lower() else 'vicuna',
            model=self.model,
            image_loader=self.image_loader,
            tokenizer=self.tokenizer,
            is_training=False
        )
        # ============================================================================
        self.options_system_prompt = (
            'Carefully read the following question and select the letter corresponding '
            'to the correct answer. Highlight the applicable choices without giving '
            'explanations.'
        )
        self.wo_options_system_prompt = (
            'Carefully read the following question Answer the question directly.'
        )
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

    def use_custom_prompt(self, dataset):
        return True
        if listinstr(['MCQ', 'VQA'], DATASET_TYPE(dataset)):
            return True
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            return True
        elif dataset is not None and listinstr(['MME'], dataset):
            return True
        return False

    def build_prompt(self, line, dataset: str):
        image_paths: list[str] = self.dump_image(line, dataset)
        prompt: str = line['question']
        if dataset == 'MME':
            # prompt = prompt + ' Answer the question using a single word or phrase.'
            messages = [dict(type='image', value=image_path) for image_path in image_paths]
            messages.append(dict(type='text', value=prompt))
            return messages
        else:
            print(line, dataset)
            raise NotImplementedError
        # if DATASET_TYPE(dataset) == 'Y/N':
        #     prompt = line['question'] + ' Please answer yes or no.'
        msgs = []
        system_prompt = ''
        tgt_path = self.dump_image(line, dataset)
        prompt = line['question']
        print(prompt)
        assert prompt.endswith('Please answer yes or no.')
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs
        raise ValueError('debug')
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt = ''

        question = line['question']
        if DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                system_prompt = self.options_system_prompt + '\nPlease just indicate your choice.'
            else:
                system_prompt = self.wo_options_system_prompt
            if 'MMMU' in dataset:  # Corner Case
                prompt = system_prompt + '\n' + prompt
                system_prompt = ''
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question'] + ' Yes or No?'
            prompt = question
        elif dataset is not None and listinstr(['MME'], dataset):
            question = line['question'] + ' Yes or No?'
            prompt = question
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            system_prompt = self.vqa_prompt
            question = line['question']
            prompt = question
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['LLaVABench', 'MMLongBench_DOC'], dataset):
                system_prompt = ''
                prompt = question
            elif listinstr(['MMVet'], dataset):
                system_prompt = self.detail_system_prompt
                prompt = question
            else:
                system_prompt = self.vqa_prompt
                prompt = question

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def generate_inner(self, messages: list[dict[str, str]], dataset: Optional[str] = None):
        # Step 1: Generation Args
        generate_kwargs = dict(
            do_sample=False,
            max_new_tokens=10,
            use_cache=False,
            pad_token_id=self.tokenizer.pad_token_id
        )
        # Step 2: Building input_tensor and image list
        image_paths = [msg['value'] for msg in messages if msg['type'] == 'image']
        prompts = [msg['value'] for msg in messages if msg['type'] == 'text']
        assert len(prompts) == 1 and len(image_paths) == 1
        dialog: list[dict[str, str]] = [dict(role='user', content='<image>' + prompts[0])]
        input_ids = self.template_applier.dialog_to_input(dialog, has_image=True)['input_ids']
        # print('--------------------------------------------------------')
        # length = len(input_tensor['input_ids'])
        # for i in range(length):
        #     print(input_tensor['input_ids'][i].item(),
        #           input_tensor['attention_mask'][i].item(),
        #           f"'{self.tokenizer.decode(input_tensor['input_ids'][i]) if input_tensor['input_ids'][i].item() != -200 else '<image>'}'", sep='\t')
        # print('--------------------------------------------------------')
        image = self.image_loader.load_image(image_paths[0])
        input_ids = input_ids.to(device='cuda').unsqueeze(0)
        images = image.to(device='cuda', dtype=self.torch_dtype).unsqueeze(0)
        with torch.inference_mode():
            image_token_pos = torch.where(input_ids == -200)[1][0].item()
            embed_tokens = self.model.get_input_embeddings()
            embeds_l = embed_tokens(input_ids[:, :image_token_pos])
            embeds_r = embed_tokens(input_ids[:, image_token_pos+1:])
            image_feature = self.model.model.encode_images(images)
            output_ids = self.model.generate(
                inputs_embeds=torch.cat([embeds_l, image_feature, embeds_r], dim=1),
                **generate_kwargs
            )
        output: str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # print(prompts[0], output)
        # input()
        return output
        prompt = ''
        image_list = []
        for message in messages:
            if message['type'] == 'image':
                image_list.append(self.image_loader(message['value']))
                prompt += '<image>'
            elif message['type'] == 'text':
                prompt += message['value']
            else:
                raise ValueError(f'Unknown message type {message["type"]}')

        print(message, dataset, type(dataset))
        raise ValueError('debug')
        if DATASET_TYPE(dataset) == 'MCQ':
            max_new_tokens = 200
        elif DATASET_TYPE(dataset) == 'Y/N':
            max_new_tokens = 3
        else:
            max_new_tokens = 1024

        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
            pad_token_id=self.pad_token_id
        )
        prompt = ""
        image = []
        for msg in message:
            if msg['type'] == 'text':
                prompt += msg['value']
            elif msg['type'] == 'image':
                image.append(self.image_loader(msg['value']))
                prompt += '\n<image>\n'
        prompt += "Please use only Yes or No to answer the question."
        assert len(image) == 1
        output_ids = self.model.chat(
            prompt=prompt.strip(),
            list_image=image,
            tokenizer=self.tokenizer,
            **generate_kwargs,
        )
        output = self.tokenizer.batch_decode(output_ids)
        return output[0]

    def chat_inner(self):
        raise NotImplementedError
