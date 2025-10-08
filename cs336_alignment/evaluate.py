import json

from typing import Callable, List, Union
from vllm import LLM, SamplingParams
from pathlib import Path

from cs336_alignment.utils import print_color, safe_slug
from cs336_alignment.data_utils import load_and_format_prompts, extract_reference_answer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def get_response(vllm_model: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
    """
    Generate responses from the language model for a list of prompts.
    """
    result = vllm_model.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text.strip() for output in result]
    return outputs


def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        cots: List[str],
        true_answers: List[str],
        eval_sampling_params: SamplingParams
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    responses = get_response(vllm_model, prompts, eval_sampling_params)
    allinfo_dict_list = []
    for response, cot, true_answer, prompt in zip(responses, cots, true_answers, prompts):
        extracted_answer = extract_reference_answer(response)
        reward_dict = reward_fn(response, cot)

        info_dict: dict[str, Union[str, float]] = {
            "prompt": prompt,
            "response": response,
            "cot": cot,
            "true_answer": true_answer,
            "extracted_answer": extracted_answer,
            **reward_dict,
        }

        allinfo_dict_list.append(info_dict)
    
    return allinfo_dict_list


def main(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    data_path: str = "./data/gsm8k/test.jsonl",
    prompt_path: str = "./cs336_alignment/prompts/r1_zero.prompt",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
):
    print_color(f"Eavluating {model_name} on {data_path}")

    vllm_model = LLM(model_name, dtype="half")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    prompts, cots, true_answers = load_and_format_prompts(data_path, prompt_path)

    results = evaluate_vllm(vllm_model, r1_zero_reward_fn, prompts, cots, true_answers, sampling_params)

    # save the results
    model_tag = safe_slug(model_name)
    data_stem = Path(data_path).stem
    out_dir = Path("evaluations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"evaluate_{model_tag}_{data_stem}.jsonl"

    correct_count = 0
    format_reward = 0
    answer_reward = 0
    reward = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for i in results:
            if i["extracted_answer"] == i["true_answer"]:
                correct_count += 1
            format_reward += i["format_reward"]
            answer_reward += i["answer_reward"]
            reward += i["reward"]
            json.dump(i, f)
            f.write("\n")
    
    print_color(f"Correct answers: {correct_count}/{len(results)}", "green")
    print_color(f"Format rewards: {format_reward}/{len(results)}", "green")
    print_color(f"Answer rewards: {answer_reward}/{len(results)}", "green")
    print_color(f"Total rewards: {reward}/{len(results)}", "green")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()