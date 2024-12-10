
import argparse

import torch
from dialogues import DialogueTemplate, get_dialogue_template
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, set_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        help="Name of model to generate samples with",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="The model repo's revision to use",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None, help="Overrides the dialogue template's system prompt"
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(42)

    prompts = [
        [{"role": "user", "content": "Please write a C/C++ code test case for testing GCC compiler bugs, with the specific directions for testing the compiler bugs as follows:friend template function declaration within template class"}],
        [{"role": "user", "content": "Please write a C/C++ code test case for testing GCC compiler bugs, with the specific directions for testing the compiler bugs as follows:assuming & on overloaded member function incorrectly reported"}],
        [{"role": "user", "content": "Please write a C/C++ code test case for testing GCC compiler bugs, with the specific directions for testing the compiler bugs as follows:Optimizer bug in gcc-2.95.2 C++"}]
    ]


    try:
        dialogue_template = DialogueTemplate.from_pretrained(args.model_id, revision=args.revision)
    except Exception:
        print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
        dialogue_template = get_dialogue_template("no_system")

    if args.system_prompt is not None:
        dialogue_template.system = args.system_prompt
    formatted_prompts = []
    for prompt in prompts:
        dialogue_template.messages = [prompt] if isinstance(prompt, dict) else prompt
        formatted_prompts.append(dialogue_template.get_inference_prompt())

    print("=== SAMPLE PROMPT ===")
    print(formatted_prompts[0])
    print("=====================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"EOS token ID for generation: {tokenizer.convert_tokens_to_ids(dialogue_template.end_token)}")
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
        min_new_tokens=32,
        max_new_tokens=256,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
    )
    outputs = ""
    for idx, prompt in enumerate(formatted_prompts):
        batch = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
        generated_ids = model.generate(**batch, generation_config=generation_config)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False).lstrip()
        outputs += generated_text + "\n\n"
        print(f"=== EXAMPLE {idx} ===")
        print()
        print(generated_text)
        print()
        print("======================")
        print()

    raw_model_name = args.model_id.split("/")[-1]
    model_name = f"{raw_model_name}"
    if args.revision is not None:
        model_name += f"-{args.revision}"

    with open(f"data/samples-{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(outputs)


if __name__ == "__main__":
    main()
