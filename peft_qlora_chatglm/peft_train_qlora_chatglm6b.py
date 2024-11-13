from typing import Dict

import datasets
import transformers
from data_collator import DataCollatorForChatGLM
from datasets import load_dataset
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer

from config.config import (compute_dtype_map, lora_alpha, lora_rank,
                           max_input_length, model_name_or_path, prompt_text,
                           seed, train_data_name_or_path)


# tokenzi_func 处理
def tokenize_func(example: Dict, tokenizer: transformers.PreTrainedTokenizer, ignore_label_id: int =-100) -> Dict:
    """
    对单个数据样本进行tokenize处理

    params:
    example (dict): 包含'content'和'summary'键的字典，代表训练数据的一个样本。
    tokenizer (transformer.PreTrainedTokenizer): tokenize文本
    ignore_label_id (int, optional): 在label中用于填充的忽略ID，默认为-100。

    return:
    dict: 包含'tokenized_input_ids'和'labels'的字典，用于模型训练。
    """

    # 构建问题文本
    question = prompt_text + example['content']
    if example.get('input', None) and example['input'].strip():
        question += f'\n{example["input"]}'

    # 构建答案文本
    answer = example['summary']

    # tokenize questions and answer
    question_ids = tokenizer.encode(text=question, add_special_tokens=False)
    answer_ids = tokenizer.encode(text=answer, add_special_tokens=False)

    # truncation
    if len(question_ids) > max_input_length - 2:
        question_ids = question_ids[:max_input_length - 2]
    if len(answer_ids) > max_input_length - 2:
        answer_ids = answer_ids[:max_input_length - 2]

    # construct model input format
    input_ids = tokenizer.build_inputs_with_special_tokens(question_ids, answer_ids)
    question_length = len(question_ids) + 2  # 加上 gmask 和 bos 标记

    # 构建标签，对于问题部分的输入使用ignore_label_id进行填充
    labels = [ignore_label_id] * question_length + input_ids[question_length:]

    return {'input_ids': input_ids, 'labels': labels}


def load_datasets(datasets_name: str, model_name: str):
    # DatasetDict({
    # train: Dataset({
    #     features: ['content', 'summary'],
    #     num_rows: 114599
    # })
    # validation: Dataset({
    #     features: ['content', 'summary'],
    #     num_rows: 1070
    # })
    # })
    dataset = load_dataset(datasets_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True,
                                              revision='b098244')

    column_names = dataset['train'].column_names
    tokenized_dataset = dataset['train'].map(
        lambda example: tokenize_func(example, tokenizer),
        batched=False,
        remove_columns=column_names
    )

    tokenized_dataset = tokenized_dataset.shuffle(seed=seed)
    tokenized_dataset = tokenized_dataset.flatten_indices()

    # 准备数据整理器
    data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id)

    return tokenized_dataset, data_collator


def load_qlora_model(model_name: str):
    # QLoRA 量化配置
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=compute_dtype_map['bf16'])

    model = AutoModel.from_pretrained(model_name,
                                      quantization_config=q_config,
                                      device_map='auto',
                                      trust_remote_code=True,
                                      revision='b098244')

    # 获取当前模型占用的 GPU显存（差值为预留给 PyTorch 的显存）
    memory_footprint_bytes = model.get_memory_footprint()
    memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)  # 转换为 MiB

    print(f"当前加载的模型占用{memory_footprint_mib:.2f}MiB")
    return model


def add_adapt_2_model(kbit_model, target_modules='chatglm'):
    lora_config = LoraConfig(
        target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[target_modules],
        r=lora_rank,
        lora_alpha=lora_alpha,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM
    )
    qlora_model = get_peft_model(kbit_model, lora_config)
    qlora_model.print_trainable_parameters()
    return qlora_model


def train(qlora_model, tokenized_dataset, data_collator):
    training_args = TrainingArguments(
        output_dir="./models/chatglm_qlora",    # 输出目录
        per_device_train_batch_size=16,           # 每个设备的训练批次
        gradient_accumulation_steps=4,            # 梯度累积步数
        # per_device_eval_batch_size=8,             # 每个设备的评估批量大小
        learning_rate=1e-3,                       # 学习率
        num_train_epochs=1,                       # 训练轮数
        lr_scheduler_type="linear",                # 学习率调度器类型
        warmup_ratio=0.1,                         # 预热比例
        logging_steps=10,                         # 日志记录步数
        save_strategy="steps",                    # 模型保存策略
        save_steps=100,                           # 模型保存步数
        # evaluation_strategy="steps"               # 评估策略
        # eval_steps=500,                           # 评估步数
        optim="adamw_torch",                      # 优化器类型
        fp16=True,                                # 是否使用混合精度训练
    )

    trainer = Trainer(
        model=qlora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    trainer.train()
    trainer.model.save_pretrained("./models/fine_tune/chatglm_qlora")


if __name__ == "__main__":
    
    tokenized_dataset, data_collator = load_datasets(train_data_name_or_path, model_name_or_path)
    model = load_qlora_model(model_name_or_path)
    kbit_model = prepare_model_for_kbit_training(model)
    qlora_model = add_adapt_2_model(kbit_model)
    train(qlora_model, tokenized_dataset, data_collator)