from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from torch.nn import functional as F
from spacy.lang.en import English
import nltk
import numpy as np
import argparse
from llm_models.modeling_gemma2 import Gemma2ForCausalLM
from llm_models.modeling_qwen2 import Qwen2ForCausalLM
from llm_models.modeling_llama import LlamaForCausalLM
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import re
import pandas as pd

# Define font configuration based on OS
system = platform.system()
if system == 'Darwin':  # macOS
    # Check if specific fonts are available on macOS
    font_list = [f.name for f in fm.fontManager.ttflist]
    
    if 'Apple Color Emoji' in font_list:
        emoji_font = 'Apple Color Emoji'
    else:
        emoji_font = None
        
    # Use macOS-specific font configuration
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 
                                 'Heiti SC', 'Microsoft YaHei', 'DejaVu Sans']
elif system == 'Linux':
    # Linux font configuration
    plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Sans CJK TC',
                                 'Noto Color Emoji', 'DejaVu Sans']
else:  # Windows
    # Windows font configuration
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 
                                 'Segoe UI Emoji', 'DejaVu Sans']

# Ensure proper Unicode handling
mpl.rcParams['axes.unicode_minus'] = False

class RAG_Detector:

    def __init__(self, model_name, eval_mode):
        self.model_name = model_name
        self.eval_mode = eval_mode
        
        self.tokenizer, self.model = self.load_model(model_name)
        nltk.download("punkt_tab")
        self.generate_kwargs = {"max_new_tokens": 512, "do_sample": False}
        self.prompt_template = "Context: {context}\n\nQuery: {query}"

    def run_detector(self, context, query):
        sample_outputs = self.context_detection(context, query)
        related_attn_mlp = self.attn_mlp_locate(sample_outputs)
        self.attn_mlp_heatmap(related_attn_mlp)
        self.mlp_ans(sample_outputs)
        self.mlp_token_heatmap_plot(f"./output/pre_mlp_layer_words.csv", f"./output/pre_mlp_layer_words_heatmap.png")
        self.mlp_token_heatmap_plot(f"./output/after_mlp_layer_words.csv", f"./output/after_mlp_layer_words_heatmap.png")

        return sample_outputs, related_attn_mlp


    def context_detection(self, context, query):

        sample_outputs = {}

        
        context_parts, separators, start_indices = self.context_partitioner(context)
        output, output_ids = self.get_response(context, query)
        full_response_logits, ablated_contexts, context_parts, JSD_div_values = self.ablate_context(context, query,
                                                                                context_parts,
                                                                                separators,
                                                                                output_ids)

        JSD_top_index = torch.argmax(torch.tensor(JSD_div_values)).item()

        sample_outputs['original_context'] = context
        sample_outputs['query'] = query
        sample_outputs['ablated_contexts'] = ablated_contexts
        sample_outputs['top_context_index'] = JSD_top_index 
        sample_outputs['output_ids'] = output_ids
        sample_outputs['response'] = output
        sample_outputs['locate_context'] = context_parts[JSD_top_index]
        sample_outputs['context_parts'] = context_parts
        sample_outputs['JSD_div_values'] = JSD_div_values
        sample_outputs['full_resposne_logits'] = full_response_logits

        
        return sample_outputs


    def model_vocab_size(self):
        # Get the model's vocabulary size
        vocab_size = self.model.vocab_size
        # print(f"Model's vocabulary size: {vocab_size}")
        return vocab_size



    def load_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if "gemma" in model_name:
            model = Gemma2ForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                use_flash_attention_2='flash_attention_2',
            ).eval()
        elif "Qwen" in model_name:
            model = Qwen2ForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                use_flash_attention_2='flash_attention_2',
                # trust_remote_code=True
            ).eval()
        elif "llama" in model_name:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                use_flash_attention_2='flash_attention_2',
            ).eval()
        return tokenizer, model

    def context_partitioner(self, context, partition_type="sentence"):

        """Split response into parts and return the parts, start indices, and separators."""
        parts = []
        separators = []
        start_indices = []

        for line in context.splitlines():
            if partition_type == "sentence":
                parts.extend(nltk.sent_tokenize(line))
            elif partition_type == "word":
                tokenizer = English().tokenizer
                parts = [token.text for token in tokenizer(context)]
            else:
                raise ValueError(f"Cannot split response by '{partition_type}'")

        cur_start = 0
        for part in parts:
            cur_end = context.find(part, cur_start)
            separator = context[cur_start:cur_end]
            separators.append(separator)
            start_indices.append(cur_end)
            cur_start = cur_end + len(part)

        return parts, separators, start_indices

    def get_ablated_context(self, parts, separators):
       ablated_contexts = []
        mask = (np.ones((len(parts), len(parts))) - np.eye(len(parts))).astype(bool)
        for row_index in range(mask.shape[0]):
            ablated_separators = np.array(separators)[mask[row_index]]
            ablated_parts = np.array(parts)[mask[row_index]]
            context = ""
            for i, (separator, part) in enumerate(zip(ablated_separators, ablated_parts)):
                if i > 0:
                    context += separator
                context += part
            ablated_contexts.append(context)
        return ablated_contexts

    def get_prompt_id(self, context, query):
        prompt = self.prompt_template.format(context=context, query=query)
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chat_prompt_ids = self.tokenizer.encode(chat_prompt, add_special_tokens=False)

        return chat_prompt_ids, self.tokenizer.decode(chat_prompt_ids)

    def get_response(self, input_context, input_query):
        
        chat_prompt_ids, chat_prompt = self.get_prompt_id(input_context, input_query)
        input_ids = torch.tensor([chat_prompt_ids], device=self.model.device)
        output_ids = self.model.generate(input_ids, **self.generate_kwargs)[0]

        
        raw_output = self.tokenizer.decode(output_ids)
        prompt_length = len(self.tokenizer.decode(chat_prompt_ids))

        
        response = raw_output[prompt_length:]
        response_logits = output_ids[prompt_length:]
        return response, self.tokenizer(response, add_special_tokens=False).input_ids,


    def ablate_context(self, input_context, input_query, input_context_parts, separators, output_ids):
        ablated_contexts = self.get_ablated_context(input_context_parts, separators)

       
        full_logits = self.full_context_logit(input_context, input_query, output_ids)

       
        JSD_div_values = []
        for i, ablated_context in enumerate(ablated_contexts):
            ablated_chat_prompt_ids, _ = self.get_prompt_id(ablated_context, input_query)
            ablated_input_ids = torch.tensor([ablated_chat_prompt_ids + output_ids], device=self.model.device)
            with torch.no_grad():
                ablated_output = self.model(input_ids=ablated_input_ids)
            ablated_logits = ablated_output.logits[:, -(len(output_ids) + 1): -1].bfloat16()

            
            jsd = self.calculate_dist_2d(ablated_logits.squeeze(0), full_logits.squeeze(0))
            JSD_div_values.append(jsd)

            
            del ablated_logits, ablated_output, ablated_input_ids
            torch.cuda.empty_cache()

        return full_logits, ablated_contexts, input_context_parts, JSD_div_values


    def get_ablated_attn_hidden_logit(self, input_contexts, top_context_index, input_query, output_ids,
                                      full_context_mode=False):

        
        if full_context_mode:
            ablated_chat_prompt_ids, ablated_chat_prompt = self.get_prompt_id(input_contexts, input_query)
        else:
            ablated_chat_prompt_ids, ablated_chat_prompt = self.get_prompt_id(input_contexts[top_context_index],
                                                                              input_query)
        ablated_input_ids = ablated_chat_prompt_ids + output_ids
        ablated_input_ids = torch.tensor([ablated_input_ids], device=self.model.device)
        
        cache_outputs = self.model(input_ids=ablated_input_ids, attn_out_fg=True, output_attentions=True)
        return cache_outputs.attentions  # [layer_index[hidden_states, cache_kv, cache_attn_out]]


    
    def logit_lens(self, input_vectors, output_ids, accumulate_mode=False):
        with torch.no_grad():
            logits = self.model.lm_head(self.model.model.norm(input_vectors)[:, -(len(output_ids) + 1): -1])
            if 'gemma' in self.model_name:
                logits = self.gemma_logit_process(logits)
        labels = torch.tensor([output_ids], device=self.model.device)
        batch_size, seq_length = labels.shape
        # [num_tokens x vocab_size]
        reshaped_logits = logits.reshape(batch_size * seq_length, -1)
        reshaped_labels = labels.reshape(batch_size * seq_length)
        correct_logits = reshaped_logits.gather(-1, reshaped_labels[:, None])[:, 0]  # [num_tokens]
        if not accumulate_mode:
            return correct_logits.sum()
        else:
            cloned_logits = reshaped_logits.clone()
            cloned_logits.scatter_(-1, reshaped_labels[:, None], -torch.inf)
            other_logits = cloned_logits.logsumexp(dim=-1)
            reshaped_outputs = correct_logits - other_logits
            return reshaped_outputs.sum()

    def locate_attn_JSD(self, ablated_cache_hiddens, full_cache_hiddens, output_ids):
        JSD_div_values = self.cal_attn_head_JSD(ablated_cache_hiddens, full_cache_hiddens, output_ids)

        # sort the JSD_div_values in descending order for all layers and all heads and for each sorted value, get the index of the layer and head

        attn_heads_JSD = []
        for layer_index in range(self.model.config.num_hidden_layers):
            for head_index in range(self.model.config.num_attention_heads):
                attn_heads_JSD.append((layer_index, head_index, JSD_div_values[layer_index][head_index].item()))
        # sort the top_attn_heads in descending order based on the JSD_div_values
        top_attn_heads = sorted(attn_heads_JSD, key=lambda x: x[2], reverse=True)
        return top_attn_heads

    def gemma_logit_process(self, logits):
        if self.model.config.final_logit_softcapping is not None:
            logits = logits / self.model.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.model.config.final_logit_softcapping
        return logits

    def cal_attn_head_JSD(self, ablated_cache_hiddens, full_context_hiddens, output_ids):
        JSD_div_values = torch.zeros((self.model.config.num_hidden_layers, self.model.config.num_attention_heads),
                                     device=self.model.device, dtype=torch.bfloat16)
        layer_index = 0
        for i in range(1, len(ablated_cache_hiddens), 2):
            layer_attn_head_lens = []
            for j in range(self.model.config.num_attention_heads):

                with torch.no_grad():
                    ablated_logits = self.model.lm_head(
                        self.model.model.norm(ablated_cache_hiddens[i][:, j, :])[:, -(len(output_ids) + 1): -1])
                    full_logits = self.model.lm_head(
                        self.model.model.norm(full_context_hiddens[i][:, j, :])[:, -(len(output_ids) + 1): -1])
                    if 'gemma' in self.model_name:
                        ablated_logits = self.gemma_logit_process(ablated_logits)
                        full_logits = self.gemma_logit_process(full_logits)
                    

                JSD_div_value = self.calculate_dist_2d(ablated_logits.squeeze(0), full_logits.squeeze(0))
                
                JSD_div_values[layer_index][j] = JSD_div_value
                
            layer_index += 1
        return JSD_div_values



    def full_context_logit(self, input_context, input_query, output_ids):

        
        ablated_contexts = [input_context]
       
        for i, ablated_context in enumerate(ablated_contexts):
            # print(f"Context {i}: {ablated_context}")

            ablated_chat_prompt_ids, ablated_chat_prompt = self.get_prompt_id(ablated_context, input_query)
            ablated_input_ids = torch.tensor([ablated_chat_prompt_ids + output_ids], device=self.model.device)
            with torch.no_grad():
                ablated_output = self.model(input_ids=ablated_input_ids)
            logits = ablated_output.logits[:, -(len(output_ids) + 1): -1]
           
        return logits.bfloat16()

    def calculate_dist_2d(self, ablated_context_logits, full_context_logits):
        assert ablated_context_logits.shape == full_context_logits.shape, "Logits must have the same shape"
        # Calculate softmax
        softmax_ablate = F.softmax(ablated_context_logits, dim=-1)
        softmax_full = F.softmax(full_context_logits, dim=-1)

        # Calculate the average distribution M
        M = 0.5 * (softmax_ablate + softmax_full)

        # Calculate log-softmax for the KL divergence
        log_softmax_ablate = F.log_softmax(ablated_context_logits, dim=-1)  # Adding epsilon for numerical stability
        log_softmax_full = F.log_softmax(full_context_logits, dim=-1)  # Adding epsilon for numerical stability

        # Calculate the KL divergences and then the JS divergences
        kl1 = F.kl_div(log_softmax_ablate, M, reduction='none').sum(dim=-1)
        kl2 = F.kl_div(log_softmax_full, M, reduction='none').sum(dim=-1)
        js_divs = 0.5 * (kl1 + kl2)

        scores = js_divs.cpu().tolist()

        return sum(scores)



    def attn_head_locate(self, original_context, ablated_contexts, top_context_index, input_query, output_ids):
        ablated_cache_outputs = self.get_ablated_attn_hidden_logit(ablated_contexts, top_context_index, input_query,
                                                                   output_ids)
        full_context_outputs = self.get_ablated_attn_hidden_logit(original_context, None, input_query, output_ids,
                                                                  full_context_mode=True)
       

        
        if self.eval_mode == "JSD":

            sort_attn_heads = self.locate_attn_JSD(ablated_cache_outputs, full_context_outputs, output_ids)
            

       
        return sort_attn_heads  # (layer_index, head_index, JSD_div_value)


    def get_ablated_mlp_logit(self, input_contexts, top_context_index, input_query, output_ids, full_context_mode=False,
                              pre_mlp_mode=False, after_mlp_res_mode=False, pre_mlp_residual_mode=False):

       
        if full_context_mode:
            ablated_chat_prompt_ids, ablated_chat_prompt = self.get_prompt_id(input_contexts, input_query)
        else:
            ablated_chat_prompt_ids, ablated_chat_prompt = self.get_prompt_id(input_contexts[top_context_index],
                                                                              input_query)
        ablated_input_ids = ablated_chat_prompt_ids + output_ids
        ablated_input_ids = torch.tensor([ablated_input_ids], device=self.model.device)
        
        if pre_mlp_mode:
            cache_outputs = self.model(input_ids=ablated_input_ids, pre_mlp_input_fg=True)
        elif after_mlp_res_mode:
            cache_outputs = self.model(input_ids=ablated_input_ids, output_hidden_states=True, after_mlp_res_fg=True)
            return cache_outputs.hidden_states  # [layer_index[after_mlp_res]]
        elif pre_mlp_residual_mode:
            cache_outputs = self.model(input_ids=ablated_input_ids, pre_mlp_residual_fg=True)
        else:
            cache_outputs = self.model(input_ids=ablated_input_ids, mlp_output_fg=True)
        return cache_outputs.attentions  # [layer_index[cache_mlp_out]]

    def cal_mlp_JSD(self, ablated_cache_mlps, full_context_mlps, output_ids):
        
        JSD_div_values = torch.zeros((self.model.config.num_hidden_layers), device=self.model.device,
                                     dtype=torch.bfloat16)
        layer_index = 0
        for i in range(len(ablated_cache_mlps)):
            # layer_attn_head_lens = []
            with torch.no_grad():
                ablated_logits = self.model.lm_head(
                    self.model.model.norm(ablated_cache_mlps[i])[:, -(len(output_ids) + 1): -1])
                full_logits = self.model.lm_head(
                    self.model.model.norm(full_context_mlps[i])[:, -(len(output_ids) + 1): -1])
                if 'gemma' in self.model_name:
                    ablated_logits = self.gemma_logit_process(ablated_logits)
                    full_logits = self.gemma_logit_process(full_logits)
                

            JSD_div_value = self.calculate_dist_2d(ablated_logits.squeeze(0), full_logits.squeeze(0))
            
            JSD_div_values[layer_index] = JSD_div_value
            
            layer_index += 1
        return JSD_div_values

    def locate_mlp(self, original_context, input_query, top_context_index, output_ids):
        ablated_mlp_logits = self.get_ablated_mlp_logit(original_context, top_context_index, input_query, output_ids)
        full_mlp_logits = self.get_ablated_mlp_logit(original_context, None, input_query, output_ids,
                                                     full_context_mode=True)

        JSD_div_values = self.cal_mlp_JSD(ablated_mlp_logits, full_mlp_logits, output_ids)

        sorted_JSD, sorted_indices = torch.sort(JSD_div_values, descending=True)
        top_mlps = list(zip(sorted_indices.float().cpu().numpy(), sorted_JSD.float().cpu().numpy()))
        sorted_mlps = [(int(layer_index), float(JSD_value)) for layer_index, JSD_value in top_mlps]

        return sorted_mlps  # (layer_index, JSD_div_value)



    def decode_mlp_layer_words(self, original_context, input_query, output_ids):
        pre_full_mlp_layers_res_logits = self.get_ablated_mlp_logit(original_context, None, input_query, output_ids,
                                                                    full_context_mode=True, pre_mlp_residual_mode=True)
        pre_full_mlp_layers_res_logits = torch.stack(pre_full_mlp_layers_res_logits, dim=0).squeeze(
            1)  # [layer_index, seq_len, vocab_size]

        after_full_mlp_layers_res_logits = self.get_ablated_mlp_logit(original_context, None, input_query, output_ids,
                                                                      full_context_mode=True, after_mlp_res_mode=True)
        after_full_mlp_layers_res_logits = torch.stack(after_full_mlp_layers_res_logits[1:], dim=0).squeeze(
            1)  # [layer_index, seq_len, vocab_size]

        if 'gemma' not in self.model_name:
            pre_full_mlp_layers_res_probs = F.softmax(self.model.lm_head(
                self.model.model.norm(pre_full_mlp_layers_res_logits)[:, -(len(output_ids) + 1): -1]), dim=-1)
            after_full_mlp_layers_res_probs = F.softmax(self.model.lm_head(
                self.model.model.norm(after_full_mlp_layers_res_logits)[:, -(len(output_ids) + 1): -1]), dim=-1)
        elif 'gemma' in self.model_name:
            pre_full_mlp_layers_res_probs = F.softmax(self.gemma_logit_process(self.model.lm_head(
                self.model.model.norm(pre_full_mlp_layers_res_logits)[:, -(len(output_ids) + 1): -1])), dim=-1)
            after_full_mlp_layers_res_probs = F.softmax(self.gemma_logit_process(self.model.lm_head(
                self.model.model.norm(after_full_mlp_layers_res_logits)[:, -(len(output_ids) + 1): -1])), dim=-1)

        # print(full_mlp_layers_probs.shape)
        pre_max_probs, pre_tokens = pre_full_mlp_layers_res_probs.max(dim=-1)
        pre_mlp_layer_words = [[self.tokenizer.decode(token.item(), skip_special_tokens=True) for token in layer_token]
                               for layer_token in pre_tokens]
        after_max_probs, after_tokens = after_full_mlp_layers_res_probs.max(dim=-1)
        after_mlp_layer_words = [
            [self.tokenizer.decode(token.item(), skip_special_tokens=True) for token in layer_token] for layer_token in
            after_tokens]
        response_words = [self.tokenizer.decode(token, skip_special_tokens=True) for token in output_ids]

        pre_mlp_layer_words_prob_pairs = [
            [(pre_mlp_layer_words[i][j], pre_max_probs[i][j].item()) for j in range(len(pre_mlp_layer_words[i]))] for i
            in range(len(pre_mlp_layer_words))]
        after_mlp_layer_words_prob_pairs = [
            [(after_mlp_layer_words[i][j], after_max_probs[i][j].item()) for j in range(len(after_mlp_layer_words[i]))]
            for i in range(len(after_mlp_layer_words))]

        # usd pandas to store pre and after as two tables, where each row is a layer, and each column is a token with the corresponding prob
        pre_mlp_layer_words_df = pd.DataFrame(pre_mlp_layer_words_prob_pairs, columns=response_words)
        after_mlp_layer_words_df = pd.DataFrame(after_mlp_layer_words_prob_pairs, columns=response_words)
        pre_mlp_layer_words_df.index = [f"Layer {i}" for i in range(len(pre_mlp_layer_words))]
        after_mlp_layer_words_df.index = [f"Layer {i}" for i in range(len(after_mlp_layer_words))]

        output_path = f"./output/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        pre_mlp_layer_words_df.to_csv(f"{output_path}pre_mlp_layer_words.csv", index=True)
        after_mlp_layer_words_df.to_csv(f"{output_path}after_mlp_layer_words.csv", index=True)



    
    def mlp_ans(self, sample_outputs):
        
        self.decode_mlp_layer_words(sample_outputs['original_context'], sample_outputs['query'], sample_outputs['output_ids'])

    def attn_mlp_locate(self, sample_outputs):

        related_attn_mlp = {}
        
        sorted_attn_heads = self.attn_head_locate(sample_outputs['original_context'],
                                                    sample_outputs['ablated_contexts'],
                                                    sample_outputs['top_context_index'],
                                                    sample_outputs['query'],
                                                    sample_outputs['output_ids'])
        related_attn_mlp['sorted_attn_heads'] = sorted_attn_heads

        

        sorted_mlp = self.locate_mlp(sample_outputs['original_context'],
                                        sample_outputs['query'],
                                        sample_outputs['top_context_index'],
                                        sample_outputs['output_ids'])
        related_attn_mlp['sorted_mlp'] = sorted_mlp
            

        return related_attn_mlp

    def attn_mlp_heatmap(self, related_attn_mlp, output_path="./output/"):
        # Determine the maximum layer and head numbers
        max_layer = 0
        max_head = 0
        # for sample_id in data:
        for layer, head, _ in related_attn_mlp['sorted_attn_heads']:
            max_layer = max(max_layer, layer)
            max_head = max(max_head, head)

        # Initialize a matrix to store the sum of JSD scores and a count matrix
        attn_sum = np.zeros((max_layer + 1, max_head + 1))
        attn_count = np.zeros((max_layer + 1, max_head + 1))

        # Sum up the JSD scores for each layer-head combination across all samples
        # for sample_id in data:
        for layer, head, jsd in related_attn_mlp['sorted_attn_heads']:
            attn_sum[layer, head] += jsd
            attn_count[layer, head] += 1

        # Calculate the average JSD score for each layer-head combination
        # Avoid division by zero by using np.divide with a 'where' condition
        attn_avg = np.divide(attn_sum, attn_count, out=np.zeros_like(attn_sum), where=attn_count!=0)

        # Transpose the matrix to swap x and y axes
        attn_avg_transposed = attn_avg.T  # This swaps layers and heads

        # Create the heatmap with smaller cells and square aspect ratio
        plt.figure(figsize=(10, 8))  # Adjust figure size for better aspect ratio
        ax = sns.heatmap(attn_avg_transposed, 
                        cmap="YlGnBu", 
                        annot=False,
                        fmt=".2f", 
                        linewidths=0.1,  # Decrease linewidths for smaller cells
                        cbar_kws={'label': 'Average JSD Score', 'shrink': 0.8})  # Add shrink parameter

        # Set the title and labels (swap x and y labels)
        plt.title('Average JSD Scores for Attention Heads', fontsize=16)
        plt.xlabel('Layer Number', fontsize=14)  # Now x-axis represents layers
        plt.ylabel('Head Number', fontsize=14)   # Now y-axis represents heads

        # Set x-ticks to display only every 6 layers (now on x-axis)
        xtick_positions = np.arange(0.5, max_layer + 1.5, 6)  # 0.5 centers in cells, step of 6
        xtick_labels = [str(int(pos - 0.5)) for pos in xtick_positions]  # Convert to layer numbers
        plt.xticks(xtick_positions, xtick_labels, rotation=0, fontsize=12)

        # Make cells appear more square by adjusting aspect ratio
        ax.set_aspect('equal')  # This makes each cell square

        # Save the figure
        plt.tight_layout()
        plt.savefig(f'{output_path}attention_head_jsd_heatmap.png', dpi=300)
        # plt.show()

        # Create a separate heatmap for MLP JSD scores
        plt.figure(figsize=(10, 8))

        # Initialize a matrix for MLP scores
        mlp_sum = np.zeros(max_layer + 1)
        mlp_count = np.zeros(max_layer + 1)

        # Sum up the MLP JSD scores for each layer across all samples
        # for sample_id in data:
        for layer, jsd in related_attn_mlp['sorted_mlp']:
            mlp_sum[layer] += jsd
            mlp_count[layer] += 1

        # Calculate the average MLP JSD score for each layer
        mlp_avg = np.divide(mlp_sum, mlp_count, out=np.zeros_like(mlp_sum), where=mlp_count!=0)

        # Create a new figure for MLP heatmap with smaller cells and better proportions
        plt.figure(figsize=(10, 2))  # Adjusted for square aspect ratio

        # Reshape for heatmap as a row vector (1 row × N layers)
        mlp_avg_reshaped = mlp_avg.reshape(1, -1)

        # Create the MLP heatmap with smaller cells
        ax_mlp = sns.heatmap(mlp_avg_reshaped, 
                        cmap="YlGnBu", 
                        annot=False,
                        fmt=".2f", 
                        linewidths=0.1,  # Same as attention heatmap
                        cbar_kws={'label': 'Average JSD Score', 'shrink': 0.8})

        # Set the title and labels
        plt.title('Average MLP JSD Scores by Layer', fontsize=16)
        plt.xlabel('Layer Number', fontsize=14)

        # Set x-ticks to display only every 6 layers (same as attention heatmap)
        xtick_positions = np.arange(0.5, max_layer + 1.5, 6)  # 0.5 centers in cells, step of 6
        xtick_labels = [str(int(pos - 0.5)) for pos in xtick_positions]  # Convert to layer numbers
        plt.xticks(xtick_positions, xtick_labels, rotation=0, fontsize=12)

        # Remove y-ticks labels
        plt.yticks([])

        # Make cells appear more square by adjusting aspect ratio
        ax_mlp.set_aspect('equal')  # This value will need adjustment based on your display

        # Save the figure
        plt.tight_layout()
        plt.savefig(f'{output_path}mlp_jsd_heatmap.png', dpi=300)

    def mlp_token_heatmap_plot(self, input_file, output_file):

        # Load the CSV file
        df = pd.read_csv(input_file)
        


        # Get column names (excluding the first column which contains layer names and the last column)
        column_names = df.columns[1:-1].tolist()  # Updated to exclude the final column

        # Extract layer numbers and create a list to store probabilities
        num_layers = len([col for col in df.index if df.iloc[col, 0].startswith('Layer ')])
        num_tokens = len(column_names)

        # Create matrices for tokens and probabilities
        token_matrix = [['' for _ in range(num_tokens)] for _ in range(num_layers)]
        prob_matrix = np.zeros((num_layers, num_tokens))

        # Function to extract token name and probability from a cell
        def extract_token_prob(cell):
            # Handle potential NaN values
            if pd.isna(cell):
                return "", 0.0
            
            # Extract using regex with more flexible pattern to handle various characters
            match = re.search(r'\(\'(.*)\', ([\d\.]+)\)', str(cell))
            if match:
                token = match.group(1)
                prob = float(match.group(2))
                return token, prob
            return "", 0.0

        # Fill the matrices (excluding the final column)
        for i in range(num_layers):
            layer_row = df[df.iloc[:,0] == f'Layer {i}']
            if not layer_row.empty:
                for j in range(num_tokens):
                    cell = layer_row.iloc[0, j+1]  # +1 to skip the 'Layer X' column
                    token, prob = extract_token_prob(cell)
                    token_matrix[i][j] = token
                    prob_matrix[i][j] = prob

        # Function to handle potentially problematic characters and truncate long text
        def clean_for_display(text):
            if not text:
                return ""
            
            # Replace certain problematic characters
            text = text.replace('\t', '⇥').replace('\n', '↵')
            
            # Truncate text longer than 7 characters with "..."
            if len(text) > 9:
                return text[:9] + "..."
            
            return text

        # Create a figure with larger dimensions to accommodate full tokens
        plt.figure(figsize=(24, 26))  # Increased figure size

        # Create the heatmap with more space between cells
        ax = sns.heatmap(prob_matrix, 
                        cmap="Blues", 
                        vmin=0, 
                        vmax=1.0,
                        cbar_kws={'label': 'Probability'},
                        linewidths=0.5,  # Add lines between cells
                        square=True)     # Make cells square

        # Set the title and labels
        plt.title('Top Token Probabilities Across Layers', fontsize=16)
        plt.xlabel('Response Tokens', fontsize=14)
        plt.ylabel('Layer Number', fontsize=14)

        # Set x-axis ticks to show the column names from the CSV
        plt.xticks(np.arange(num_tokens) + 0.5, column_names, rotation=0, ha='center', fontsize=13)

        # Set y-axis ticks to show layer numbers
        plt.yticks(np.arange(num_layers) + 0.5, [f'{i}' for i in range(num_layers)], fontsize=11)

        # Adjust the font size based on token length
        for i in range(num_layers):
            for j in range(num_tokens):
                # Only show tokens with high probability to avoid cluttering
                if prob_matrix[i][j] > 0.0:  # Show all tokens
                    # Clean the text and truncate if needed
                    display_text = clean_for_display(token_matrix[i][j])
                    
                    # Adjust text color based on background
                    text_color = 'white' if prob_matrix[i][j] > 0.7 else 'black'
                    
                    # Adjust font size (now simpler since all texts are at most 10 chars)
                    font_size = 13
                        
                    try:
                        plt.text(j + 0.5, i + 0.5, display_text, 
                                ha='center', va='center',
                                fontsize=font_size,
                                color=text_color,
                                wrap=True,
                                family=plt.rcParams['font.family'][0])
                    except:
                        # If still failing, use a simple indicator
                        plt.text(j + 0.5, i + 0.5, '•', 
                                ha='center', va='center',
                                fontsize=10, color=text_color)

        # Add more padding around the plot
        plt.tight_layout(pad=2.0)

        # Save the figure with higher resolution
        plt.savefig(output_file, dpi=400, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--eval_mode", type=str, default="JSD")
    args = parser.parse_args()

   

    context = """
    Attributing Response to Context: A Jensen–Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation

    Abstract
    Retrieval-Augmented Generation (RAG) leverages large language models (LLMs) combined with external contexts to enhance the accuracy and reliability of generated responses. However, reliably attributing generated content to specific context segments, context attribution, remains challenging due to the computationally intensive nature of current methods, which often require extensive fine-tuning or human annotation. In this work, we introduce a novel Jensen–Shannon Divergence driven method to Attribute Response to Context (ARC-JSD), enabling efficient and accurate identification of essential context sentences without additional fine-tuning or surrogate modelling. Evaluations on a wide range of RAG benchmarks, such as TyDi QA, Hotpot QA, and Musique, using instruction-tuned LLMs in different scales demonstrate superior accuracy and significant computational efficiency improvements compared to the previous surrogate-based method. Furthermore, our mechanistic analysis reveals specific attention heads and multilayer perceptron (MLP) layers responsible for context attribution, providing valuable insights into the internal workings of RAG models.
    """
    query = "Which datasets are used in this paper?"

    rag_detector = RAG_Detector(args.model_name, args.eval_mode)
    rag_detector.run_detector(context, query)



