import contextlib
import json
import os
import sys
from abc import abstractmethod
from contextlib import contextmanager
from time import time
from typing import Dict, List, Optional, Tuple

import ray
import torch
from accelerate import Accelerator  # type: ignore
from ray.air import session
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

import trlx.utils.logging as logging
from trlx.data.configs import TRLConfig
from trlx.pipeline import MiniBatchIterator
from trlx.trainer import BaseRLTrainer, register_trainer
from trlx.utils import (
    filter_non_scalars,
    get_distributed_config,
    get_git_tag,
    get_optimizer_class,
    get_scheduler_class,
    significant,
)
from trlx.utils.modeling import (
    flatten_dict,
    freeze_bottom_causal_layers,
    freeze_bottom_seq2seq_layers,
    gather_dict,
    get_delta_model_class,
    parse_delta_kwargs,
)

import json
import os
import uuid
from time import time
from typing import Callable, List

import ray
from ray.air import session
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from trlx.pipeline import MiniBatchIterator

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.models.modeling_ppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.utils import Clock, infinite_dataloader, significant
from rich.console import Console
from rich.table import Table
from trlx.utils.modeling import RunningMoments, gather_dict, logprobs_of_labels

logger = logging.get_logger(__name__)


@register_trainer
class AcceleratePPOAutoTFMTrainer(AcceleratePPOTrainer):
    """PPO AutoTFM Accelerate Trainer"""

    def __init__(self, config: TRLConfig, **kwargs):
        """PPO AutoTFM Accelerate Trainer initialization

        Args:
            config: Config
        """
        super().__init__(config, **kwargs)

    def generate_ref_model(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)
        if self.generate_experience_kwargs is not None:
            kwargs = dict(self.generate_experience_kwargs, **kwargs)
        else:
            kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.accelerator.unwrap_model(self.ref_model).generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def generate_eval_ref_model(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)

        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.accelerator.unwrap_model(self.ref_model).generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )


    def learn(self):  # noqa: C901
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """
        logger.info("Starting training")

        self.prepare_learning()
        self.iter_count = 0
        self.nth_evaluation = 0

        if ray.is_initialized():
            print("Ray is initialized. Print out the evaluation before training\n=========\n")
            checkpoint = session.get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as dir:
                    self.accelerator.load_state(dir)

                    with open(os.path.join(dir, "state.json")) as f:
                        state = json.load(f)
                        self.iter_count = state["iter_count"]
        else:
            print("Ray is NOT initialized. START Print out the evaluation before training\n=========\n")
            results = self.evaluate()
            self.accelerator.log(results, step=self.iter_count)

            results_ref_model = self.evaluate_ref_model()
            self.accelerator.log(results_ref_model, step=self.iter_count)
            print("Ray is NOT initialized. DONE Print out the evaluation before training\n=========\n")

        tbar = logging.tqdm(
            initial=self.iter_count,
            total=self.total_steps,
            disable=not self.accelerator.is_local_main_process,
            position=0,
            leave=True,
        )

        best_reward = -float("inf")

        epoc = 0 # epoch counter
        # For each epoch
        for _ in range(self.config.train.epochs):
            epoc += 1
            # For each batch
            mbc = 0  # mini batch counter
            for mbs in MiniBatchIterator(self.train_dataloader, self.mb_size, self.num_mb):
                mbc += 1
                # For each update per batch
                nubc = 0  # number of updates per batch
                for _ in range(self.n_updates_per_batch):
                    nubc += 1
                    # Note that whereas standard policy gradient methods perform one
                    # gradient update per batch, PPO for example commonly performs
                    # multiple gradient updates on the same batch of data.
                    # https://arxiv.org/pdf/1707.06347.pdf
                    forward_time = 0
                    backward_time = 0
                    stats_accum = []
                    for mb in mbs:
                        with self._accumulate():
                            forward_time -= time()
                            loss, stats = self.loss(mb)
                            forward_time += time()
                            backward_time -= time()
                            self.accelerator.backward(loss)
                            backward_time += time()
                            stats_accum.append(stats)

                    print(
                        f"\nAfter forward/backward updates, epoc:{epoc}, iter_count: {self.iter_count}, mini batch counter: {mbc}, number update per batch: {nubc}")
                    forward_time /= self.num_mb
                    backward_time /= self.num_mb
                    # TODO(Dahoas): Best way to combine stats between mbs?
                    # How does accelerate do it?
                    stats = {key: sum([stats[key] for stats in stats_accum]) / self.num_mb for key in stats_accum[0]}

                    self.opt.step()
                    self.opt.zero_grad()
                    self.scheduler.step()
                    self.iter_count += 1
                    print(f"Done updating optimizer and scheduler.step(), iter_count: {self.iter_count}, total_steps: {self.total_steps}")

                    if (
                        self.iter_count % self.config.train.checkpoint_interval == 0
                        or self.iter_count >= self.total_steps
                    ):
                        print(f"Checkpointing, epoc:{epoc}, iter_count: {self.iter_count}, mini batch counter: {mbc}, number update per batch: {nubc}")
                        subfolder = f"checkpoint_{self.iter_count:0{len(str(self.total_steps))}d}"
                        directory = os.path.join(self.config.train.checkpoint_dir, subfolder)
                        logger.info(f"Saving intermediate checkpoint into {directory}")
                        if self.config.train.save_optimizer:
                            self.save(directory)
                        else:
                            self.save_pretrained(directory)

                    stats["time/forward"] = forward_time
                    stats["time/backward"] = backward_time
                    for group_number, lr in enumerate(self.scheduler.get_last_lr()):
                        stats[f"learning_rate_group_{group_number}"] = lr

                    if self.iter_count % self.config.train.eval_interval == 0 or self.iter_count >= self.total_steps:
                        results = self.evaluate()
                        results_ref_model = self.evaluate_ref_model()
                        stats.update(results)
                        print(f"Evaluation, epoc:{epoc}, iter_count: {self.iter_count}, total_steps: {self.total_steps}, mini batch counter: {mbc}, number update per batch: {nubc}")
                        print(f"Evaluation, best reward: {best_reward}, epoc:{epoc}, config.train.eval_interval: {self.config.train.eval_interval}")

                        if ray.is_initialized():
                            session.report(filter_non_scalars(stats), checkpoint=checkpoint)

                        # always save checkpoint with the greatest mean reward
                        if self.config.train.save_best:
                            if stats.get("reward/mean", -float("inf")) > best_reward:
                                best_reward = stats.get("reward/mean")
                                do_save = True
                            # in case ILQL reports reward estimate as one of its metrics
                            elif stats.get("metrics/reward", -float("inf")) > best_reward:
                                best_reward = stats.get("metrics/reward")
                                do_save = True
                            else:
                                do_save = False
                            do_save = torch.tensor(do_save, device=self.accelerator.device)
                            if torch.distributed.is_initialized():
                                torch.distributed.all_reduce(do_save, torch.distributed.ReduceOp.MAX)
                            if do_save:
                                directory = os.path.join(self.config.train.checkpoint_dir, "best_checkpoint")
                                logger.info(f"Saving the best state so far into {directory}")
                                if self.config.train.save_optimizer:
                                    self.save(directory)
                                else:
                                    self.save_pretrained(directory)

                    desc = " | ".join(f"{k}: {v:.2f}" for k, v in stats.items() if k.startswith("loss"))
                    tbar.set_description(f"[{desc}]")
                    tbar.update()

                    self.accelerator.log(stats, step=self.iter_count)

                    if self.iter_count >= self.total_steps:
                        print(f"Return results. iter_count > total_steps, iter_count: {self.iter_count}, total_steps: {self.total_steps}")
                        return results

                self.post_backward_callback()

            self.post_epoch_callback()
        tbar.close()

    def evaluate_ref_model(self):  # noqa: C901
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        logger.info("Evaluating ref_model")
        print("Evaluating ref_model")

        # Do multiple evaluations over a single list in `gen_kwargs` if present
        if self.generate_sweep_kwarg is not None:
            gen_sweep_arg, gen_sweep_values = self.generate_sweep_kwarg
        else:
            gen_sweep_values = [None]

        desc = [
            f"generation sweep 0/{len(gen_sweep_values)}",
            f"eval batch 0/{len(self.eval_dataloader)}",
        ]
        tbar = logging.tqdm(
            total=len(self.eval_dataloader) * len(gen_sweep_values),
            desc=f"[{' | '.join(desc)}]",
            disable=not self.accelerator.is_main_process,
            position=0,
            leave=True,
        )

        stats = {}
        table = []

        for i_sweep, gen_sweep_value in enumerate(gen_sweep_values):
            # A dedicated suffix for wandb logging
            if gen_sweep_value is not None:
                sweep_suffix = f"@{gen_sweep_arg}={gen_sweep_value}"
            else:
                sweep_suffix = ""

            all_samples = []
            all_prompts = []
            all_prompt_sizes = []
            all_metadata = []
            generate_time = time()
            for i_prompt, prompts in enumerate(self.eval_dataloader):
                metadata = {k: v for k, v in prompts.items() if k != "input_ids" and k != "attention_mask"}
                if self.generate_sweep_kwarg:
                    samples = self.generate_eval_ref_model(
                        prompts["input_ids"], prompts["attention_mask"], **{gen_sweep_arg: gen_sweep_value}
                    )
                else:
                    samples = self.generate_eval_ref_model(prompts["input_ids"], prompts["attention_mask"])

                # TODO(reciprocated): this should be moved into `decode`
                # but that needs to be synced with indexing in `make_experience`
                if self.config.model.model_arch_type == "seq2seq":
                    samples = samples[:, 1:].contiguous()

                prompt_sizes = torch.tensor(prompts.input_ids.shape[1]).repeat(len(prompts.input_ids))
                prompts, samples, prompt_sizes = self.accelerator.gather_for_metrics(
                    self.accelerator.pad_across_processes(
                        [prompts.input_ids, samples, prompt_sizes.to(samples.device)],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                    )
                )
                all_samples.extend(samples.tolist())
                all_prompts.extend(prompts.tolist())
                all_prompt_sizes.extend(prompt_sizes.tolist())

                metadata = gather_dict(metadata, self.accelerator.gradient_state)
                all_metadata.append(metadata)

                desc = [
                    f"generation sweep {i_sweep + 1}/{len(gen_sweep_values)}",
                    f"eval batch {i_prompt + 1}/{len(self.eval_dataloader)}",
                ]
                tbar.set_description(f"[{' | '.join(desc)}]")
                tbar.update()
            tbar.close()

            stats["time/generate"] = time() - generate_time

            if self.accelerator.is_main_process:
                str_samples, str_prompts, str_outputs = self.decode(all_prompts, all_samples, all_prompt_sizes)

                columns = ["prompt", "output"]
                columns_data = [str_prompts, str_outputs]

                metadata, *xs = all_metadata
                for k in metadata:
                    for x in xs:
                        metadata[k].extend(x[k])

                # in online setting, compute the reward for validation
                if self.reward_fn:
                    logger.info("Computing ref_model rewards")
                    rewards = torch.tensor(
                        self.reward_fn(
                            samples=str_samples, 
                            prompts=str_prompts, 
                            outputs=str_outputs, 
                            mode="ref_eval",
                            **metadata),
                        dtype=float,
                    )
                    mean_reward = rewards.mean().item()
                    columns.append("ref reward")
                    if not isinstance(rewards, list):
                        rewards = rewards.tolist()
                    columns_data.append(rewards)
                    stats[f"reward/mean{sweep_suffix}"] = mean_reward

                # additionally log any other metrics
                if self.metric_fn:
                    logger.info("Computing ref_model metrics")
                    metric_time = time()
                    metrics = self.metric_fn(samples=str_samples, prompts=str_prompts, outputs=str_outputs, **metadata)
                    stats["time/metric"] = time() - metric_time

                    mean_metrics = {
                        f"metrics/{k}{sweep_suffix}": torch.as_tensor(xs).mean(-1).item() for k, xs in metrics.items()
                    }

                    stats.update(mean_metrics)

                    for metric, values in metrics.items():
                        # Skip metrics that are scalers since they represent aggregated values
                        if isinstance(values, float):
                            continue
                        columns.append(metric)
                        if not isinstance(values, list):
                            values = values.tolist()
                        columns_data.append(values)

                # Prepend the sweep argument along with samples
                if self.generate_sweep_kwarg:
                    columns.insert(0, gen_sweep_arg)
                    columns_data.insert(0, [gen_sweep_value] * len(samples))

                table.append(list(zip(*columns_data)))

        # Log and display evaluation metrics
        logger.info("Summarizing ref_model evaluation")
        if self.accelerator.is_main_process:
            rows = sum(list(map(list, zip(*table))), [])

            # Add metrics/rewards to the table's title
            table_title = f"ref_model, Evaluation #{self.nth_evaluation}"
            for k, x in stats.items():
                if k.startswith("reward") or k.startswith("metrics"):
                    table_title += f" {k}: {significant(x)}"

            rich_table = Table(*columns, title=table_title, show_lines=True)
            for ix in range(max(min(3, len(rows)), len(gen_sweep_values))):
                rich_table.add_row(*[str(significant(x)) for x in rows[ix]])
            Console().print(rich_table)

            if self.config.train.tracker == "wandb":
                import wandb

                stats["samples"] = wandb.Table(columns, rows)

        self.nth_evaluation += 1
        return stats

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

        clock = Clock()
        ppo_rl_elements = []
        accumulated_stats = []

        while len(ppo_rl_elements) < num_rollouts:
            stats = {}
            # Get next batch in prompt dataset
            batch: PromptBatch = next(self.prompt_iterator)

            rollout_generate_time = time()

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            samples = self.generate(batch["input_ids"], batch["attention_mask"], do_sample=False)
            # samples = self.generate(batch["input_ids"], batch["attention_mask"])
            stats["time/rollout_generate"] = time() - rollout_generate_time

            prompt_tensors = batch.input_ids
            device = samples.device

            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
            padded_samples = self.accelerator.pad_across_processes(
                samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )

            padded_prompts = self.accelerator.pad_across_processes(
                prompt_tensors, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)
            metadata = gather_dict({k: v for k, v in batch.items() if k != "input_ids" and k != "attention_mask"})

            #ref_samples = self.generate_ref_model(batch["input_ids"], batch["attention_mask"], do_sample=False)
            #ref_samples = self.generate_ref_model(batch["input_ids"], batch["attention_mask"])
            #padded_ref_samples = self.accelerator.pad_across_processes(
            #    ref_samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            #)
            #gathered_ref_samples = self.accelerator.gather(padded_ref_samples)

            if self.accelerator.is_main_process:
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True
                )

                #all_str_ref_samples, all_str_ref_prompts, all_str_ref_outputs = self.decode(
                #    gathered_prompts, gathered_ref_samples, gathered_prompt_sizes, append_eos_token=True
                #)

                rollout_score_time = time()
                all_scores = torch.tensor(
                    self.reward_fn(
                        samples=all_str_samples,
                        prompts=all_str_prompts,
                        outputs=all_str_outputs,
                        mode="train",
                        #ref_prompts=all_str_ref_prompts,
                        #ref_samples=all_str_ref_samples,
                        #ref_outputs=all_str_ref_outputs,
                        **metadata
                    ),
                    dtype=torch.float,
                    device=device,
                )
                stats["time/rollout_score"] = time() - rollout_score_time

                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1).unbind())
            else:
                all_scores = None

            if torch.distributed.is_initialized():
                scores = torch.empty(len(samples), device=device)
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = all_scores[0].clone().detach()

            str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples, append_eos_token=True)

            # Pad the sample outputs
            outputs = self.tokenizer(str_outputs).input_ids
            if self.config.model.model_arch_type == "seq2seq":
                # add <pad> to the start of the output
                for i in range(len(outputs)):
                    outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]

            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            sample_outputs = torch.vstack(outputs).to(device)

            if self.config.method.cliprange_reward:
                scores = torch.clip(scores, -self.config.method.cliprange_reward, self.config.method.cliprange_reward)

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running_moments.update(scores)
            stats["rollout_scores/mean"] = all_scores_mean.item()
            stats["rollout_scores/std"] = all_scores_std.item()
            stats["rollout_scores/running_mean"] = self.running_moments.mean.item()
            stats["rollout_scores/running_std"] = self.running_moments.std.item()

            if self.config.method.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == "ref":
                scores /= self.ref_std

            # Precompute logprobs, values
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = batch.attention_mask.to(device)
                prompt_tensors = batch.input_ids.to(device)
                decoder_attention_mask = sample_outputs.not_equal(self.tokenizer.pad_token_id)
                decoder_attention_mask[:, 0] = 1
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=prompt_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=sample_outputs,
                        decoder_attention_mask=decoder_attention_mask,
                    )
                    logits = outputs.logits
                    values = outputs.value
                    if hasattr(self.model, "frozen_head"):
                        ref_logits = self.model.forward_hydra(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
            else:
                all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
                attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                with torch.no_grad():
                    logits, *_, values = self.model(
                        all_tokens,
                        attention_mask=attention_mask,
                    )
                    # TODO(dahoas): When hydra model works need to also support generation on hydra head
                    if hasattr(self.model, "frozen_head"):
                        ref_logits = self.model.forward_hydra(
                            all_tokens,
                            attention_mask=attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            all_tokens,
                            attention_mask=attention_mask,
                            return_dict=True,
                        ).logits
                        ref_logits = ref_logits.to(device)

            if self.config.model.model_arch_type == "seq2seq":
                logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], sample_outputs[:, 1:])
            else:
                logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

            n_samples: int = samples.shape[0]

            # Estimate the KL divergence between the model and reference model
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = sample_outputs != self.tokenizer.pad_token_id
                start = 0
            else:
                start = prompt_tensors.shape[1] - 1

            log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
            kl = log_ratio.exp() - 1 - log_ratio
            mean_kl_per_token = kl.mean()
            mean_kl = kl.sum(1).mean()

            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            values = values.cpu()[:, :-1]

            # Get the logprobs and values, for tokens that are not padding,
            # from the start of the prompt up to the <eos> token, while also including the latter
            # (these are taken from the student model and not the reference model)
            ends = start + attention_mask[:, start:].sum(1) + 1
            all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
            all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]

            kl_penalty = self.kl_ctl.value * -log_ratio.cpu()
            kl_penalty = [xs[start : ends[ix]] for ix, xs in enumerate(kl_penalty)]

            rollout_count = 0

            for sample_idx in range(n_samples):
                rewards = kl_penalty[sample_idx]
                rewards[-1] += scores[sample_idx].cpu()

                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=prompt_tensors[sample_idx],
                        response_tensor=sample_outputs[sample_idx],
                        logprobs=all_logprobs[sample_idx],
                        values=all_values[sample_idx],
                        rewards=rewards,
                    )
                )

                rollout_count += 1

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_kl, torch.distributed.ReduceOp.AVG)

            stats["time/rollout_time"] = clock.tick()
            stats["policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
            stats["policy/kl_per_token"] = torch.sqrt(mean_kl_per_token).item()
            accumulated_stats.append(stats)

            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in stats}
        stats["kl_ctl_value"] = self.kl_ctl.value
        self.mean_kl = stats["policy/sqrt_kl"] ** 2
        self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)

