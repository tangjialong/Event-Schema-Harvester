import os
import stat
import time
import nvgpu
import argparse
import gradio as gr

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_max_memo():
    gpus = nvgpu.available_gpus()
    max_memo = {}
    for gpu in gpus:
        gpu = int(gpu)
        max_memo[gpu] = "78000MiB"
    return max_memo
    
def get_model(model_name):
    model2path = {
        "bloom": "model/bloom",
    }
    model_path = model2path[model_name]
    
    max_memo = get_max_memo()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", max_memory=max_memo)
    model.eval()
    
    return tokenizer, model

def predict(text, length, do_sample=False, num_return_sequences=1, num_beams=1):
    model_inputs = tokenizer(text, return_tensors="pt").input_ids
    generate_ids = model.generate(
        model_inputs, 
        max_length=length, 
        no_repeat_ngram_size=5,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams
    )
    output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
    output_text = "\n\n###\n\n".join(output_text)
    print(output_text)
    return output_text

def predict_by_file(
    input_type, length, input_text, 
    file_obj, output_type, out_filename, 
    do_sample, num_return_sequences, num_beams
):
    length = int(length)
    
    if length > 256:
        return "too long"
    
    num_return_sequences = int(num_return_sequences)
    num_beams = int(num_beams)
    if not do_sample:
        num_return_sequences=1
    
    if input_type == "text":
        return predict(input_text, length, do_sample=do_sample, num_return_sequences=num_return_sequences, num_beams=num_beams)
    else:
        if file_obj is None or out_filename is None:
            return "need a file"
    
    t = time.strftime("_%m_%d_%H_%M", time.localtime())
    out_filename += t
    
    input_file = file_obj.name
    with open(input_file, "r") as f:
        texts = f.read().split("\n\n")
    f.close()
    
    outputs = []
    if output_type == "file":
        f_out = open("bloom_demo/output_files/{}".format(out_filename), "w", 1)
    for text in texts:
        try:
            output_text = predict(text, length, do_sample=do_sample, num_return_sequences=num_return_sequences, num_beams=num_beams)
        except RuntimeError as err:
            print(err)
            continue
        else:
            outputs.append(output_text)

        if output_type == "file":
            f_out.write(output_text)
            f_out.write("\n\n###\n\n")
    if output_type == "file":
        os.chmod("bloom_demo/output_files/{}".format(out_filename), stat.S_IROTH)
        return "/shared_home/tangjialong/Projects/bloom_demo/output_files/" + out_filename
    else:
        return "\n\n###\n\n".join(outputs)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--model-name", type=str, default="bloom")
    parser.add_argument("--demo-type", type=str, default="file")
    args = parser.parse_args()

    print("Loading model...")
    tokenizer, model = get_model(args.model_name)
    print("Model loaded!")
    
    if args.demo_type == "text":
        demo = gr.Interface(
            fn=predict,
            inputs=["text", "text"],
            outputs=["text"],
        )
    elif args.demo_type == "file":
        demo = gr.Interface(
            fn=predict_by_file,
            inputs=[
                gr.Radio(["text", "file"]),
                gr.Number(),
                "text",
                "file",
                gr.Radio(["text", "file"]),
                "text",
                gr.Checkbox(label="do_sample"),
                gr.inputs.Slider(1, 10, label="num_return_sequences", default=1, step=1),
                gr.inputs.Slider(1, 10, label="num_beams", default=1, step=1)
            ],
            outputs=["text"]
        )
    demo.launch(share=True, enable_queue=True)
