import torch
import os
from hydra.bert.src.bert_layers import BertForMaskedLM
from recformer import RecmambaConfig, RecmambaForPretraining
from hydra.bert.src.configuration_bert import BertConfig as HydraBertConfig


# Create BertConfig instance from pretrained model with custom parameters
hydra_config_params = {
    'num_hidden_layers': 23,
    'max_position_embeddings': 128,  # Fixed typo: was 'max_position_embedding'
    'use_position_embeddings': False,
    'hidden_size': 768,
}

# Create config using from_pretrained and update with custom parameters
hydra_config = HydraBertConfig.from_pretrained('bert-base-uncased', **hydra_config_params)

# Padding for divisibility by 8 (following create_bert.py pattern)
if hydra_config.vocab_size % 8 != 0:
    hydra_config.vocab_size += 8 - (hydra_config.vocab_size % 8)

hydra_model = BertForMaskedLM(hydra_config)
hydra_model = BertForMaskedLM.from_composer(
    pretrained_checkpoint='hydra_bert_23layers.pt',
    config=hydra_config
)


config = RecmambaConfig.from_pretrained('bert-base-uncased', **hydra_config_params)
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51
config.attention_window = [64] * 12
config.vocab_size += 8 - (config.vocab_size % 8)
model = RecmambaForPretraining(config)

## hydra weight를 불러와서, recformer weight로 저장
hydra_state_dict = hydra_model.state_dict()
recformer_state_dict = model.state_dict()
for name, param in hydra_state_dict.items():
    if name not in recformer_state_dict:
        print('missing name', name)
        continue
    else:
        try:
            if not recformer_state_dict[name].size()==param.size():
                print(name)
                print(recformer_state_dict[name].size())
                print(param.size())
                if(name == 'bert.embeddings.token_type_embeddings.weight'):
                    print("!!!FOUND!!!!")
                    hydra_state_dict[name] = param[0, :].unsqueeze(0)
                    param = hydra_state_dict[name]
                    print('after:',param.size())
            recformer_state_dict[name].copy_(param)
        except:
            print('wrong size', name)

for name, param in hydra_state_dict.items():
    if not torch.all(param == recformer_state_dict[name]):
        print(name)

if not os.path.exists('hydra_ckpt'):
    os.mkdir('hydra_ckpt')
torch.save(recformer_state_dict, 'hydra_ckpt/bert-base-uncased.bin')
