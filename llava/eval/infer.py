import argparse
import copy
import torch
import warnings
from decord import VideoReader, cpu
import numpy as np
import json
import multiprocessing as mp
import os
from multiprocessing import Pool
import functools
import itertools
import random
from tqdm import tqdm

from ERF.patch_infer import maybe_build_engine, run_erf_sample

try:  # pragma: no cover
    from icecream import ic

    ic.configureOutput(prefix="[qwen] ")
except ImportError:  # pragma: no cover
    def ic(*args, **kwargs):
        print(*args)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import rank0_print

# Prefer flash / mem-efficient attention kernels when the GPU stack supports them; otherwise fall back.
if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
    flash_available = getattr(torch.backends.cuda, "flash_sdp_available", lambda: False)()
    mem_available = getattr(torch.backends.cuda, "mem_efficient_sdp_available", lambda: False)()
    try:
        if flash_available or mem_available:
            torch.backends.cuda.enable_flash_sdp(flash_available)
            torch.backends.cuda.enable_mem_efficient_sdp(mem_available)
            torch.backends.cuda.enable_math_sdp(True)
            print(
                "[SDP] flash_sdp_available:", flash_available,
                "mem_efficient_sdp_available:", mem_available,
                "-> using flash:", torch.backends.cuda.flash_sdp_enabled(),
                "mem:", torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math:", torch.backends.cuda.math_sdp_enabled(),
            )
        else:
            raise RuntimeError("Flash/mem-efficient SDP not available")
    except Exception:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[SDP] Falling back to math kernels (flash/mem-efficient unavailable).")


warnings.filterwarnings("ignore")

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()


def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

def get_options_letter(len_options):
    if len_options==2:
        return '(A or B)'
    elif len_options==3:
        return '(A, B or C)'
    elif len_options==4:
        return '(A, B, C or D)'
    elif len_options==5:
        return '(A, B, C, D, or E)'
    else:
        raise NotImplementedError

def get_prompt(dataset_name, sample, conv_template="qwen_1_5", video_time=None, num_frames=None, frame_time=None):
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    if video_time:
        prompt = f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time}.\n"
    else:
        prompt = ""

    if dataset_name in ['VSI']:
        prompt += "These are frames of a video.\n"
        prompt += sample["question"] + "\n"
        if 'candidates' in sample:
            for op in sample["candidates"]:
                prompt += f"{op}\n"
            prompt += "Answer with the option's letter from the given choices directly."
        else:
            prompt += "Please answer the question using a single word or phrase."
    elif dataset_name in ['MovieChat']:
        if video_time is None:
            prompt += "These are frames of a video.\n"
        if 'time' in sample:
            timestamp = round(sample['time']/sample['fps'], 2)
            prompt += f"At time {timestamp}s, "
        prompt += sample["question"] + "\n"
        prompt += "Please answer the question using a single word, phrase, or sentence."
        #prompt += "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
    else:
        options_letter = get_options_letter(len(sample['candidates']))
        prompt += f"Select the best answer to the following multiple-choice question based on the video. Respond with only the letter {options_letter} of the correct option.\n"
        prompt += sample["question"] + "\n"
        for op in sample["candidates"]:
            prompt += f"{op}\n"
        prompt += f"The best answer is:"
        
    question = DEFAULT_IMAGE_TOKEN + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def run(rank, world_size, args):
    torch.cuda.set_device(rank)

    rank0_print("Loadind dataset from", args.data_path)
    with open(args.data_path, "r") as f:
        dataset = json.load(f)
     
    random.shuffle(dataset)

    num_samples = int(len(dataset) * args.test_ratio)
    dataset = dataset[rank:num_samples:world_size]
    rank0_print(f"Total samples: {num_samples}")
    print(f"Samples in rank {rank}: {len(dataset)}")

    device_map = "auto"
    if args.multiprocess or world_size > 1:
        device_map = {"": torch.device(f"cuda:{rank}")}

    overwrite_cfg = None
    if isinstance(args.temporal_pooling, int) and args.temporal_pooling and args.temporal_pooling > 1:
        overwrite_cfg = {"temporal_pooling": args.temporal_pooling}

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=args.model_name,
        lora_alpha=args.lora_alpha,
        torch_dtype="bfloat16",
        device_map=device_map,
        overwrite_config=overwrite_cfg,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    first_param = next(model.parameters())
    model_device = first_param.device
    model_dtype = first_param.dtype

    erf_engine = maybe_build_engine(args, tokenizer, model, image_processor)


    result_list = []
    for cnt, sample in enumerate(tqdm(dataset)):
        sample_save_path = f"{args.results_dir}/outputs/{sample['id']}.json"
        load_cached = os.path.exists(sample_save_path) and not args.erf_enable
        if load_cached:
            with open(sample_save_path, "r") as f:
                sample = json.load(f)
        else:
            video_path = os.path.join(args.video_root, sample["video"])
            video_np, frame_time, video_time = load_video(
                video_path, args.max_frames_num, fps=1, force_sample=True
            )
            video_tensor = image_processor.preprocess(video_np, return_tensors="pt")[
                "pixel_values"
            ]
            video_tensor = video_tensor.to(device=model_device, dtype=model_dtype)
            video_list = [video_tensor]

            if erf_engine is not None:
                erf_artifacts = run_erf_sample(
                    erf_engine,
                    sample=sample,
                    dataset_name=args.dataset_name,
                    base_prompt_builder=get_prompt,
                    max_frames_num=args.max_frames_num,
                    video_time=video_time,
                    frame_time=frame_time,
                    video_tensor=video_tensor,
                    results_dir=args.results_dir,
                )
                if erf_artifacts and erf_artifacts.get("rounds"):
                    final_round = erf_artifacts["rounds"][-1]
                    weights = final_round.get("weights", [])
                    best_idx = erf_artifacts.get("best_index", 0)
                    cand_logs = final_round.get("cand_logs") or []
                    best_log = cand_logs[best_idx] if 0 <= best_idx < len(cand_logs) else None
                    cons_dbg = best_log["score"]["cons"] if best_log else 0.0
                    evid_dbg = best_log.get("evid_sim", 0.0) if best_log else 0.0
                    cal_dbg = best_log["score"].get("cal", 0.0) if best_log else 0.0
                    weights_str = ",".join(f"{w:.2f}" for w in weights)
                    print(
                        f"[ERF] id={sample.get('id')} best={erf_artifacts.get('final_prediction')} "
                        f"w=[{weights_str}] cons={cons_dbg:.2f} "
                        f"evid={evid_dbg:.2f} cal={cal_dbg:.2f}"
                    )
            else:
                if args.use_time_ins:
                    prompt_question = get_prompt(
                        args.dataset_name,
                        sample,
                        video_time=video_time,
                        num_frames=args.max_frames_num,
                        frame_time=frame_time,
                    )
                else:
                    prompt_question = get_prompt(args.dataset_name, sample)

                input_ids = tokenizer_image_token(
                    prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(model_device)

                try:
                    with torch.inference_mode():
                        cont = model.generate(
                            input_ids,
                            images=video_list,
                            modalities=["video"],
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=False,
                        )
                except RuntimeError as exc:
                    if "no kernel found" in str(exc).lower() or "cutlass" in str(exc).lower():
                        torch.backends.cuda.enable_flash_sdp(False)
                        torch.backends.cuda.enable_mem_efficient_sdp(False)
                        torch.backends.cuda.enable_math_sdp(True)
                        torch.cuda.empty_cache()
                        with torch.inference_mode():
                            cont = model.generate(
                                input_ids,
                                images=video_list,
                                modalities=["video"],
                                do_sample=False,
                                temperature=0,
                                max_new_tokens=args.max_new_tokens,
                                use_cache=False,
                            )
                    else:
                        raise
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                sample["prediction"] = text_outputs

                del input_ids, cont

            del video_tensor, video_list, video_np
            torch.cuda.empty_cache()

            with open(sample_save_path, "w") as f:
                json.dump(sample, f, indent=4)

        result_list.append(sample)
        gt = sample.get("answer")
        score = None
        if gt is not None:
            score = 1 if fuzzy_matching(sample["prediction"]) == gt else 0
        ic({
            "idx": cnt,
            "id": sample.get("id"),
            "gt": gt,
            "prediction": sample.get("prediction"),
            "score": score,
        })
        if gt is None:
            print(cnt, "Pred:", sample["prediction"])
    
    return result_list


def main():
    parser = argparse.ArgumentParser(description="Run Inference")

    # Model
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--max_frames_num", type=int, default=16)
    # Use values >1 to pool across the temporal dimension; 0 keeps the model default.
    parser.add_argument("--temporal_pooling", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--conv_template", type=str, default="qwen_1_5")
    parser.add_argument("--use_time_ins", action="store_true")
    parser.add_argument("--lora_alpha", type=int, default=None)

    # Data
    parser.add_argument("--dataset_name", type=str, default="VideoMME")
    parser.add_argument("--data_path", type=str, default="/mnt/bum/mmiemon/datasets/Video-MME/formatted_dataset.json")
    parser.add_argument("--video_root", type=str, default="/mnt/bum/mmiemon/datasets/Video-MME/videos/data")
    parser.add_argument("--results_dir", type=str, default="/mnt/bum/mmiemon/LLaVA-NeXT/results/llava_video/VideoMME")
    parser.add_argument("--test_ratio", type=float, default=1)
    parser.add_argument("--multiprocess", action="store_true")
    parser.add_argument("--cals_acc", action="store_true")
    parser.add_argument("--erf_enable", action="store_true")
    parser.add_argument("--erf_K", type=int, default=6)
    parser.add_argument("--erf_rounds", type=int, default=2)
    parser.add_argument("--erf_tau", type=float, default=0.8)
    parser.add_argument("--erf_weights", type=str, default="0.4,0.4,0.2,0.0")
    parser.add_argument("--erf_debug", action="store_true")

    args = parser.parse_args()
    if args.model_base == "None":
        args.model_base = None

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/outputs", exist_ok=True)


    if args.multiprocess:
        mp.set_start_method("spawn")
        print(f"started benchmarking")
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        print("World size", world_size)
        with Pool(world_size) as pool:
            func = functools.partial(run, args=args, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        print("finished running")
        result_list = [res for res in itertools.chain(*result_lists)]
    else:
        result_list = run(0, world_size=1, args=args)
    

    if args.cals_acc:
        results = {"all": {"correct": 0, "total": 0}}
        for sample in result_list:
            if "answer" not in sample:
                continue
            results["all"]["total"] += 1
            if "question_type" in sample:
                if sample["question_type"] not in results:
                    results[sample["question_type"]] = {"correct": 0, "total": 0}
                results[sample["question_type"]]["total"] += 1
                
            if sample["answer"].lower()==fuzzy_matching(sample["prediction"]).lower():
                results["all"]["correct"] += 1
                if "question_type" in sample:
                    results[sample["question_type"]]["correct"] += 1

        for key in results:
            results[key]["accuracy"] = results[key]["correct"] / results[key]["total"]

        print(results)

        with open(os.path.join(args.results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
