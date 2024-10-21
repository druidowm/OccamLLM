import json
from tqdm import tqdm
from functools import partial

import numpy as np

import torch
from torch.utils.data import DataLoader

from occam_llm.models.occam_llm import OccamLLMOutput
from occam_llm.parsing import parse_numbers
from .data import NumericDataset, pad_collate_fn, prepare_synth_data, prepare_router_data

import openai


def visualize_router(raw_dataset, model, tokenizer, pad_id, num_inputs, mask_num, append_bos, use_chat, system_prompt=None, assistant_prompt=None):
    data = prepare_router_data(raw_dataset, tokenizer, num_inputs=num_inputs, mask_num=mask_num, append_bos=append_bos, use_chat=use_chat, system_prompt=system_prompt, assistant_prompt=assistant_prompt)
    dataset = NumericDataset(data)
    dataloader = DataLoader(dataset, collate_fn=partial(pad_collate_fn,pad_id=pad_id,mask_num=mask_num), batch_size=1)           
    for model_inputs, texts, results in dataloader:
        model_inputs = {k: v.to(model.llm.device) for k, v in model_inputs.items()}
        router_decision = model(**model_inputs).router_decision.argmax(-1)[0]
        input_ids = model_inputs["input_ids"][0]
        route_to_occam = torch.where(router_decision == 1)[0]
        text = ""
        last_idx = 0
        for idx in route_to_occam:
            text += tokenizer.decode(input_ids[last_idx:idx], skip_special_tokens=False) + " [OCCAM]"
            last_idx = idx
        text += tokenizer.decode(input_ids[last_idx:], skip_special_tokens=False)
        print(text + "\n#########################\n")


def eval_occam_llm(raw_dataset, model, tokenizer, pad_id, num_inputs, mask_num, append_bos, use_chat, system_prompt=None, assistant_prompt=None, do_print=False):
    data = prepare_synth_data(raw_dataset, tokenizer, num_inputs=num_inputs, mask_num=mask_num, append_bos=append_bos, use_chat=use_chat, system_prompt=system_prompt, assistant_prompt=assistant_prompt)
    dataset = NumericDataset(data)
    dataloader = DataLoader(dataset, collate_fn=partial(pad_collate_fn,pad_id=pad_id,mask_num=mask_num), batch_size=1)           
    val_results = []
    for model_inputs, texts, results in dataloader:
        model_inputs = {k: v.to(model.llm.device) for k, v in model_inputs.items()}
        outputs: OccamLLMOutput = model(**model_inputs)

        pred = outputs.pred_nums[0,-1].item()
        if abs(results[0]) < 1.e-4:
            rel_error = abs(pred)
            right = abs(pred) < 1.e-4
        else:
            rel_error = abs(results[0] - pred) / (abs(results[0]))
            right = abs(results[0] - pred) / (abs(results[0])) < 1.e-4

        val_results.append({"input": texts[0], "output": pred, "op": outputs.occamnet_ops[0][-1], "correct_answer": results[0], "is_correct": int(right), "rel_error": rel_error})
        if do_print:
            print("INPUT: " + texts[0] + "\nOUTPUT: " + "{:.4f}".format(pred) + f"  :  {outputs.occamnet_ops[0][-1]}" + "\nCORRECT ANSWER: " + "{:.8f}".format(results[0]) + f" {'(RIGHT)'  if right else '(WRONG)'}" + "\nRELATIVE ERROR: " + "{:.6f}%".format(rel_error*100))
    return val_results

def query_llm(input, llm, tokenizer, system_prompt=None, assistant_prompt=None, max_new_tokens=10):
    # Ensure they are str
    system_prompt = system_prompt or ""
    assistant_prompt = assistant_prompt or ""

    # Create dialogue
    chat = [   
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input},
    ]
 
    # Call GPT API
    if llm == "gpt-3.5-turbo" or llm == "gpt-4-turbo" or llm == "gpt-4o":
        secrets = json.load(open('secrets.json'))
        client = openai.OpenAI(
            api_key=secrets['openai-api-key'],
        )
        response = client.chat.completions.create(
            model=llm,
            messages=chat,
        )
        return response.choices[0].message.content, input
    elif llm == "gpt-4o-code" or llm == "gpt-3.5-turbo-code":
        secrets = json.load(open('secrets.json'))
        client = openai.OpenAI(
            api_key=secrets['openai-api-key'],
        )

        assistant = client.beta.assistants.create(
            name="Math Solver",
            instructions=f"Write and run code to answer math questions. {system_prompt}",
            tools=[{"type": "code_interpreter"}],
            model=llm[:-5],
        )

        thread = client.beta.threads.create(
            messages = [   
                {"role": "user", "content": input},
            ]
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            #instructions="Please address the user as Jane Doe. The user has a premium account."
        )
        if run.status == "completed":
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )

            return messages.data[0].content[0].text.value, input, run.usage.model_dump()
        return "RUN FAILED", input, None
    # Generate with OccamLLM
    elif llm.__class__.__name__ == "OccamLLM":
        response, _, input = llm.generate_text(
                        input_text=input, 
                        max_new_tokens=max_new_tokens,
                        assistant_prompt=assistant_prompt,
                        system_prompt=system_prompt,
                        temperature=0.6,
                        top_p=0.9,
                        use_cache=False,
                    )
        return "".join(response), input, None
    
    # Format in appropraite chat template for Llama
    input_ids = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(llm.device)
    # Add optional beginning of response
    if assistant_prompt != "":
        input_ids = torch.cat((input_ids, tokenizer(assistant_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(llm.device)),dim=1)
    
    # Generation ends with end of seq or end of turn tokens
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Generate
    outputs = llm.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # Decode response
    response = outputs[0][input_ids.shape[-1]:]

    response = tokenizer.decode(response, skip_special_tokens=False)
    input = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    return response, input, None


def check_output(result, response, print_results=True):

    is_correct = False
    min_rel_error = float("inf")

    if isinstance(response, int) or isinstance(response, float):
        is_correct = abs(result-response) < 10 ** -5

        try:
            min_rel_error = abs(response-result)/abs(result)
        except ZeroDivisionError:
            if is_correct:
                min_rel_error = 0
            else:
                min_rel_error = float("inf")

        return is_correct, min_rel_error
        

    
    # Parse all numbers in the response
    pred_numbers = parse_numbers(response, as_float=False)

    for nn in pred_numbers:
        # Round result to same number of decimal places as output number 
        nn_parts = nn.split(".")
        if len(nn_parts) == 2:
            dec_places = min(max(len(nn_parts[1]),2),5)
        else:
            dec_places = 2

        # Check if prediction and answer coincide up to the number of decimal palces provided by the LLM
        is_correct = is_correct or abs(result-float(nn)) < 10 ** -dec_places

        if is_correct:
            try:
                min_rel_error = abs(float(nn)-result)/abs(result)
            except ZeroDivisionError:
                min_rel_error = 0
            return is_correct, min_rel_error

        try:
            min_rel_error = min(abs(float(nn)-result)/abs(result), min_rel_error)
        except ZeroDivisionError:
            if print_results:
                print("Zero division error")
        
    return is_correct, min_rel_error


def eval_llm(dataset, llm, tokenizer, system_prompt=None, assistant_prompt=None, max_new_tokens=100, num_datapoints = 100000, print_output=False,):
    val_results = []

    np.random.seed(0)
    permuation = np.random.permutation(len(dataset))

    upto = min(num_datapoints, len(dataset))

    for index in tqdm(range(upto)):
        ex = dataset[permuation[index]]

        response, input, metadata = query_llm(ex["input"], llm, tokenizer, system_prompt=system_prompt, assistant_prompt=assistant_prompt, max_new_tokens=max_new_tokens)
        is_correct, min_rel_error = check_output(ex["output"], response)  
        
        val_results.append({"system": system_prompt, "input": input, "output": response, "correct_answer": ex["output"], "is_correct": int(is_correct), "rel_error": min_rel_error, "metadata": metadata})
        if print_output:
            print("SYSTEM: " + str(system_prompt) + "\nINPUT: " + input + "\nOUTPUT: " + response + "\nCORRECT ANSWER: " + "{:.4f}".format(ex["output"]) + f" {'(RIGHT)'  if is_correct else '(WRONG)'}" + "\nRELATIVE ERROR: " + "{:.6f}%".format(min_rel_error*100))
    return val_results
    