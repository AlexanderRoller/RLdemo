#!/usr/bin/env python3
"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array

from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from train import HumanoidWalkingTask, Model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a trained HumanoidWalkingTask checkpoint and export it as a .kinfer model"
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to the .bin checkpoint file")
    parser.add_argument("output_path", type=str, help="Path where the .kinfer model will be written")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    # --- Load task & model ---
    task: HumanoidWalkingTask = HumanoidWalkingTask.load_task(ckpt_path)
    model: Model = task.load_ckpt(ckpt_path, part="model")[0]

    # --- Gather MuJoCo joint info ---
    mujoco_model = task.get_mujoco_model()
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # drop the root joint
    num_joints = len(joint_names)

    # --- Define carry shape for RNN hidden state ---
    carry_shape: Sequence[int] = (task.config.depth, task.config.hidden_size)

    # --- Build metadata for init & step exports ---
    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=3,               # accept 3-dim control_vector commands
        carry_size=list(carry_shape),
    )

    # --- JIT-compile init function (returns initial carry) ---
    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_shape)

    # --- JIT-compile step function (sensors + time + command + carry → action + next carry) ---
    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        projected_gravity: Array,
        accelerometer: Array,
        gyroscope: Array,
        time: Array,
        command: Array,  # 3-dim control vector
        carry: Array,
    ) -> tuple[Array, Array]:
        # build observation vector (ignoring `command` if policy doesn't use it)
        obs = jnp.concatenate(
            [
                jnp.sin(time),
                jnp.cos(time),
                joint_angles,
                joint_angular_velocities,
                projected_gravity,
                accelerometer,
                gyroscope,
            ],
            axis=-1,
        )
        dist, next_carry = model.actor.forward(obs, carry)
        return dist.mode(), next_carry

    # --- Export to ONNX/KInfer intermediate format ---
    init_onnx = export_fn(init_fn, metadata)
    step_onnx = export_fn(step_fn, metadata)

    # --- Pack everything into a single .kinfer blob ---
    kinfer_model = pack(init_onnx, step_onnx, metadata)

    # --- Write out the .kinfer model ---
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(kinfer_model)

    print(f"✅ Model successfully exported to {out_path.resolve()}")


if __name__ == "__main__":
    main()

