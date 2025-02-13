import copy
import os
import sys
import time
import traceback
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
from tqdm import tqdm

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

path = os.path.dirname(found_dotenv)
print(f"Adding {path} to sys.path")
sys.path.append(path)


def get_prompt_templates(debug=False):
    system_prompt = "You are a helpful assistant." # Always respond with a JSON object containing the text."

    user_prompt = "Rewrite the following text to improve clarity, readability, and engagement while maintaining all original details and factual accuracy. Use concise, natural phrasing, and enhance flow without omitting any information. Ensure that the structure is logical, sentences are varied in length, and transitions are smooth. The tone should remain neutral and professional, suitable for a news article or formal report. Avoid redundancy while keeping all key details intact.\n\n{input}\n\n"
    if debug:
        print(f"System Prompt: {system_prompt}")
        print(f"User Prompt: {user_prompt}")

    return (
        system_prompt,
        user_prompt,
    )


def prepare_dataset(data_path, max_entries=0):
    datasets = load_dataset(
        "csv",
        data_files={"test": data_path},
    )

    if max_entries > 0:
        print(f"--- evaluating {max_entries} entries")
        ds2 = (
            copy.deepcopy(datasets["test"])
            if len(datasets["test"]) < max_entries
            else datasets["test"]
        )

        while len(ds2) < max_entries:
            ds2 = concatenate_datasets([ds2, datasets["test"]])

        datasets["test"] = Dataset.from_pandas(
            ds2.select(range(max_entries)).to_pandas().reset_index(drop=True)
        )

    print(datasets)
    return datasets


def save_results(model_name, results_path, dataset, predictions, debug=False):
    if debug:
        print(f"Saving results to: {results_path}")
    if not os.path.exists(results_path):
        # Get the directory part of the file path
        dir_path = os.path.dirname(results_path)

        # Create all directories in the path (if they don't exist)
        os.makedirs(dir_path, exist_ok=True)
        df = dataset.to_pandas()
        df.drop(columns=["text", "prompt"], inplace=True, errors="ignore")
    else:
        df = pd.read_csv(results_path, on_bad_lines="warn")

    df[model_name] = predictions

    if debug:
        print(df.head(1))

    df.to_csv(results_path, index=False)


def on_num_shots_step_completed(
    model_name, dataset, output_column, predictions, results_path
):
    save_results(
        model_name,
        results_path,
        dataset,
        predictions,
        debug=False,
    )


def invoke_openai_api(
    system_prompt,
    user_prompt,
    input,
    max_tokens=None,
    model="gpt-4o-mini",
    base_url=None,
    api_key=None,
    debug=False,
):
    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(input=input)},
    ]
    if debug:
        print(f"\nInvoking Model: {model}")
        print(f"Messages: {messages}")

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                # response_format={"type": "json_object"},
            )

            result = {"content": response.choices[0].message.content}

            if hasattr(response.choices[0].message, "reasoning_content"):
                result["reasoning_content"] = response.choices[
                    0
                ].message.reasoning_content
            else:
                parts = result["content"].split("</think>")
                if len(parts) > 1:
                    result["content"] = parts[1].strip()
                    result["reasoning_content"] = (
                        parts[0].replace("<think>", "").strip()
                    )

            result["retries"] = retries
            break
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()
            result = {"content": "Error: " + str(e)}
            retries += 1

    if debug:
        print(f"Result: {result}")

    return result


def eval_dataset_using_openai_api(
    system_prompt,
    user_prompt,
    eval_dataset,
    input_column,
    model="gpt-4o-mini",
    max_tokens=8192,
    base_url=None,
    api_key=None,
    debug=False,
):
    if debug:
        print("base_url:", base_url)
        print("api_key:", api_key[-4:])

    total = len(eval_dataset)
    predictions = []

    for i in tqdm(range(total)):
        output = invoke_openai_api(
            system_prompt,
            user_prompt,
            eval_dataset[input_column][i],
            model=model,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
            debug=i == 0 and debug,
        )
        predictions.append(output)

    return predictions


def evaluate_model_with_num_shots(
    model_name,
    data_path,
    results_path,
    range_num_shots=[0],
    start_num_shots=0,
    end_num_shots=50,
    max_entries=0,
    input_column="Text",
    output_column="Review-sentiment",
    result_column_name=None,
    debug=False,
):
    print(f"Evaluating model: {model_name}")

    datasets = prepare_dataset(data_path, max_entries=max_entries)
    # print_row_details(datasets["test"].to_pandas())

    for num_shots in range_num_shots:
        if num_shots < start_num_shots:
            continue
        if num_shots > end_num_shots:
            break

        print(f"* Evaluating with num_shots: {num_shots}")

        system_prompt, user_prompt = get_prompt_templates(debug=debug)

        start_time = time.time()  # Start time

        openai_compatible = not (
            model_name.startswith("gpt") or model_name.startswith("o")
        )
        predictions = eval_dataset_using_openai_api(
            system_prompt,
            user_prompt,
            datasets["test"],
            input_column,
            model=model_name,
            base_url=(os.getenv("BASE_URL") if openai_compatible else None),
            api_key=(
                os.getenv("DEEPSEEK_API_KEY")
                if openai_compatible and "DEEPSEEK_API_KEY" in os.environ
                else os.environ.get("OPENAI_API_KEY")
            ),
            debug=debug,
        )

        end_time = time.time()  # End time
        exec_time = end_time - start_time  # Execution time
        print(f"*** Execution time for num_shots {num_shots}: {exec_time:.2f} seconds")

        model_name_with_shots = (
            result_column_name
            if result_column_name
            else f"{model_name}/shots-{num_shots:02d}({exec_time / len(datasets['test']):.3f})"
        )

        try:
            on_num_shots_step_completed(
                model_name_with_shots,
                datasets["test"],
                output_column,
                predictions,
                results_path,
            )
        except Exception as e:
            print(e)


if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME")
    data_path = os.getenv("DATA_PATH")
    results_path = os.getenv("RESULTS_PATH")
    start_num_shots = int(os.getenv("START_NUM_SHOTS", 0))
    end_num_shots = int(os.getenv("END_NUM_SHOTS", 50))
    max_entries = int(os.getenv("MAX_ENTRIES", 0))
    output_column = os.getenv("OUTPUT_COLUMN", "Review-sentiment")

    print(
        model_name,
        data_path,
        results_path,
        start_num_shots,
        end_num_shots,
    )

    evaluate_model_with_num_shots(
        model_name,
        data_path,
        results_path=results_path,
        input_column="full_content",
        output_column=output_column,
        start_num_shots=start_num_shots,
        end_num_shots=end_num_shots,
        max_entries=max_entries,
        debug=max_entries > 0,
    )
