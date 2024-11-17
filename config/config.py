import torch


# 模型
model_name_or_path = "THUDM/chatglm3-6b"
# 训练数据
train_data_name_or_path = "HasturOfficial/adgen"
# 随机种子
seed = 8
# 输入的最大长度
max_input_length = 512
# 输出的最大长度
max_output_length = 1536
# LoRA秩
lora_rank = 4
# LoRA alpha值
lora_alpha = 32  
# LoRA Dropout率                         
lora_dropout = 0.05
# 如果从checkpoint恢复训练，指定路径
resume_from_checkpoint = None
# 所有数据前的指令文本
prompt_text = ''
# 计算数据类型（fp32, fp16, bf16）                          
compute_dtype = 'fp32'



compute_dtype_map = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
}


peft_model_path = "../models/fine_tune/chatglm_qlora"

base_model = "gpt-3.5-turbo"
base_api_key = "your api key"
base_url = "https://api.agicto.cn/v1"