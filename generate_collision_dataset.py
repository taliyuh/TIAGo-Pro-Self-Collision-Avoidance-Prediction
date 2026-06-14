#!/usr/bin/env python3

import argparse
import csv
import math
import random
import time

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState


LEFT_JOINTS = [f"arm_left_{i}_joint" for i in range(1, 8)]
RIGHT_JOINTS = [f"arm_right_{i}_joint" for i in range(1, 8)]

EXTRA_JOINTS = [
    "torso_lift_joint",
    "head_1_joint",
    "head_2_joint",
]

JOINT_NAMES = LEFT_JOINTS + RIGHT_JOINTS + EXTRA_JOINTS


def deg(x):
    return math.radians(x)


LEFT_LIMITS_DEG = [
    (-280, 150),
    (-140, 0),
    (-160, 160),
    (-140, 0),
    (-100, 100),
    (-108, 108),
    (-150, 150),
]

RIGHT_LIMITS_DEG = [
    (-40, 270),
    (-140, 0),
    (-160, 160),
    (-140, 0),
    (-220, -20),
    (-108, 108),
    (-150, 150),
]

ARM_LIMITS = [(deg(lo), deg(hi)) for lo, hi in (LEFT_LIMITS_DEG + RIGHT_LIMITS_DEG)]

EXTRA_LIMITS = [
    (0.0, 0.35),          # torso_lift_joint
    (-1.309, 1.309),      # head_1_joint
    (-1.0472, 0.34907),   # head_2_joint
]

LIMITS = ARM_LIMITS + EXTRA_LIMITS


class CollisionDatasetGenerator(Node):
    def __init__(self):
        super().__init__("collision_dataset_generator")
        self.client = self.create_client(GetStateValidity, "/check_state_validity")

    def check_collision(self, positions):
        req = GetStateValidity.Request()
        req.group_name = "both_arms_torso"

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = JOINT_NAMES
        js.position = positions

        state = RobotState()
        state.joint_state = js
        state.is_diff = True
        req.robot_state = state

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        res = future.result()
        if res is None:
            raise RuntimeError("No response from /check_state_validity")

        return int(not res.valid), len(res.contacts)

    def sample_positions(self):
        return [random.uniform(lo, hi) for lo, hi in LIMITS]

    def generate(self, output_path, n_samples, seed):
        random.seed(seed)

        self.get_logger().info("Waiting for /check_state_validity...")
        self.client.wait_for_service()
        self.get_logger().info("Service available. Generating dataset...")

        start = time.time()
        collisions = 0

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(JOINT_NAMES + ["collision", "num_contacts"])

            for i in range(n_samples):
                positions = self.sample_positions()
                collision, num_contacts = self.check_collision(positions)
                collisions += collision

                writer.writerow(
                    [f"{x:.8f}" for x in positions] + [collision, num_contacts]
                )

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    self.get_logger().info(
                        f"{i + 1}/{n_samples} samples | "
                        f"collision ratio={collisions / (i + 1):.3f} | "
                        f"{rate:.1f} samples/s"
                    )

        elapsed = time.time() - start
        self.get_logger().info(
            f"Done. Saved {n_samples} samples to {output_path}. "
            f"Collision ratio={collisions / n_samples:.3f}. "
            f"Elapsed={elapsed:.1f}s."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="/home/user/exchange/tiago_collision_dataset_17dof.csv",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rclpy.init()
    node = CollisionDatasetGenerator()

    try:
        node.generate(args.output, args.samples, args.seed)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
